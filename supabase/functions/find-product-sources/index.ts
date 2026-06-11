// Supabase Edge Function: find-product-sources
//
// POST { id: number, table: 'bom_items' | 'inventory' }
//
// For the given row, searches each of the 7 online stores for the item's
// name, then makes ONE combined Groq (Llama) call to verify which (if any)
// search result from each store is the same product. Stores with no
// confident match (plus Amazon, which has no direct search adapter) are then
// looked up via DuckDuckGo, with a second Groq call to verify those results.
// On a confident, in-stock match, upserts {source, buy_url, price, image,
// description} into the row's `sources` jsonb array.

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

// Strip any stray non-ASCII/control characters (e.g. BOM, smart quotes,
// trailing newlines from copy-paste) that would break HTTP header encoding.
const GROQ_API_KEY = (Deno.env.get('GROQ_API_KEY') ?? '').replace(/[^\x20-\x7E]/g, '').trim();
const GROQ_MODEL = Deno.env.get('GROQ_MODEL') || 'llama-3.3-70b-versatile';
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const CONFIDENCE_THRESHOLD = 0.6;
const CANDIDATE_LIMIT = 5;
const SEARCH_TIMEOUT_MS = 10000;
const GROQ_TIMEOUT_MS = 25000;

interface Candidate {
  title: string;
  price: number | null;
  url: string;
  image: string;
  description: string;
  inStock: boolean;
}

async function fetchWithTimeout(url: string, opts: RequestInit = {}, timeoutMs = SEARCH_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...opts,
      signal: controller.signal,
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36', ...(opts.headers || {}) },
    });
  } finally {
    clearTimeout(timer);
  }
}

function stripHtml(html: string): string {
  return (html || '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
}

function parsePrice(s: string): number | null {
  const m = s.replace(/,/g, '').match(/[\d.]+/);
  return m ? parseFloat(m[0]) : null;
}

function normalizeUrl(base: string, src: string): string {
  if (!src) return '';
  src = src.replace('{width}', '600');
  if (src.startsWith('//')) return `https:${src}`;
  if (src.startsWith('http')) return src;
  return `${base}${src}`;
}

// ── Shopify adapter (circuits-elec.com, store.fut-electronics.com) ──────────
// Uses Shopify's predictive-search JSON endpoint: it returns clean,
// already-absolute image URLs and each product's "body" (description), which
// is passed to Groq alongside the title for matching.
function shopifyAdapter(base: string) {
  return async (q: string): Promise<Candidate[]> => {
    const url = `${base}/search/suggest.json?q=${encodeURIComponent(q)}&resources[type]=product&resources[limit]=${CANDIDATE_LIMIT}`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const data = await res.json();
    const products = data?.resources?.results?.products;
    if (!Array.isArray(products)) return [];
    return products.map((p: any) => ({
      title: p.title || '',
      price: p.price != null ? parsePrice(String(p.price)) : null,
      url: normalizeUrl(base, (p.url || '').split('?')[0]),
      image: p.image || '',
      description: stripHtml(p.body || '').slice(0, 500),
      inStock: p.available !== false,
    }));
  };
}

// ── WooCommerce adapter (HD Electronics, microohm-eg.com, Ampere, uge-one) ──
function wooAdapter(base: string) {
  return async (q: string): Promise<Candidate[]> => {
    const url = `${base}/wp-json/wc/store/v1/products?search=${encodeURIComponent(q)}&per_page=${CANDIDATE_LIMIT}`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const data = await res.json();
    if (!Array.isArray(data)) return [];
    return data.map((p: any) => {
      const minor = p.prices?.currency_minor_unit ?? 2;
      const raw = parseFloat(p.prices?.price ?? '');
      return {
        title: p.name || '',
        price: isNaN(raw) ? null : raw / Math.pow(10, minor),
        url: p.permalink || '',
        image: p.images?.[0]?.src || '',
        description: stripHtml(p.short_description || p.description || '').slice(0, 500),
        inStock: p.is_in_stock !== false && p.is_purchasable !== false,
      };
    });
  };
}

// ── Odoo adapter (RAM Electronics) ──────────────────────────────────────────
// No reliable in-stock signal is exposed on the search results page, so
// candidates are always treated as in-stock (best-effort/limitation).
function odooAdapter(base: string) {
  return async (q: string): Promise<Candidate[]> => {
    const url = `${base}/website/search?search=${encodeURIComponent(q)}`;
    const res = await fetchWithTimeout(url);
    if (!res.ok) return [];
    const html = await res.text();
    const marker = '<a class="dropdown-item p-2 text-wrap" href="';
    const idxs: number[] = [];
    let pos = 0;
    while (true) {
      const i = html.indexOf(marker, pos);
      if (i === -1) break;
      idxs.push(i);
      pos = i + marker.length;
    }
    const out: Candidate[] = [];
    for (let k = 0; k < idxs.length && out.length < CANDIDATE_LIMIT; k++) {
      const chunkStart = idxs[k] + marker.length;
      const chunk = html.slice(chunkStart, idxs[k + 1] ?? chunkStart + 2000);
      const hrefEnd = chunk.indexOf('"');
      const href = chunk.slice(0, hrefEnd);
      if (href.includes('/category/')) continue;
      const priceM = chunk.match(/oe_currency_value">([\d.,]+)</);
      if (!priceM) continue;
      const titleM = chunk.match(/class="h6 fw-bold[^"]*">([\s\S]*?)<\/div>/);
      const title = titleM ? stripHtml(titleM[1]) : '';
      if (!title) continue;
      const imgM = chunk.match(/<img[^>]*src="([^"]+)"/);
      // Search result cards list the product's category tags as buttons;
      // surface them as a "description" so Groq has more than just the
      // title to match against.
      const categories = [...chunk.matchAll(/<button class="btn btn-link btn-sm p-0" onclick="location\.href='\/shop\/category\/[^']*'[^>]*>([^<]+)<\/button>/g)]
        .map((m) => stripHtml(m[1]));
      out.push({
        title,
        price: parsePrice(priceM[1]),
        url: href.startsWith('http') ? href : `${base}${href}`,
        image: imgM ? normalizeUrl(base, imgM[1]) : '',
        description: categories.length ? `Category: ${categories.join(', ')}` : '',
        inStock: true,
      });
    }
    return out;
  };
}

// store key (matches `source` in bom_portal.html STORES) -> adapter
const STORE_ADAPTERS: Record<string, (q: string) => Promise<Candidate[]>> = {
  'circuits-elec.com':  shopifyAdapter('https://circuits-elec.com'),
  'future-electronics': shopifyAdapter('https://store.fut-electronics.com'),
  'HD Electronics':     wooAdapter('https://hdelectronicseg.com'),
  'microohm-eg.com':    wooAdapter('https://microohm-eg.com'),
  'Ampere Electronics': wooAdapter('https://ampere-electronics.com'),
  'RAM Electronics':    odooAdapter('https://www.ram-e-shop.com'),
  'uge-one.com':        wooAdapter('https://uge-one.com'),
};

// store key -> domain/label, used for the DuckDuckGo/Amazon fallback search
const STORE_INFO: Record<string, { domain: string; label: string }> = {
  'circuits-elec.com':  { domain: 'circuits-elec.com',         label: 'Circuits-Elec' },
  'future-electronics': { domain: 'store.fut-electronics.com', label: 'Future Electronics' },
  'HD Electronics':     { domain: 'hdelectronicseg.com',       label: 'HD Electronics' },
  'microohm-eg.com':    { domain: 'microohm-eg.com',           label: 'Microohm' },
  'Ampere Electronics': { domain: 'ampere-electronics.com',    label: 'Ampere Electronics' },
  'RAM Electronics':    { domain: 'www.ram-e-shop.com',        label: 'RAM Electronics' },
  'uge-one.com':        { domain: 'uge-one.com',               label: 'UGE Electronics' },
  'Amazon':             { domain: 'amazon.eg',                 label: 'Amazon Egypt' },
};

// ── Groq (Llama) helper ──────────────────────────────────────────────────
async function callGroqJson(prompt: string): Promise<any> {
  const res = await fetchWithTimeout(
    GROQ_URL,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${GROQ_API_KEY}` },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' },
        temperature: 0,
      }),
    },
    GROQ_TIMEOUT_MS,
  );
  if (!res.ok) throw new Error(`Groq API error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  let text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error('Groq returned no content');
  text = text.trim();
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) text = fenceMatch[1].trim();
  return JSON.parse(text);
}

// ── Verification: ONE Groq call covering every store's candidates ──────────
// Reused both for the direct-store candidates and (separately) for the
// DuckDuckGo fallback candidates.
async function verifyAllMatches(
  itemName: string,
  context: string,
  candidatesByStore: Record<string, Candidate[]>,
): Promise<Record<string, { matchIndex: number | null; confidence: number; description: string }>> {
  const sections = Object.entries(candidatesByStore)
    .map(([storeKey, candidates]) => {
      const list = candidates
        .map((c, i) => `  ${i}. "${c.title}"${c.description ? ` — ${c.description.slice(0, 200)}` : ''}`)
        .join('\n');
      return `Store "${storeKey}":\n${list}`;
    })
    .join('\n\n');

  const prompt = `You are verifying whether store search results are the SAME product as a part needed for a hobby electronics / 3D-printing project.

Item needed: "${itemName}"${context ? `\nContext: ${context}` : ''}

For each store below, its numbered search results are listed:

${sections}

For EACH store listed above, pick the result that is the SAME product (same component, value and type), not just similar or related. If none of a store's results are clearly the same product, set matchIndex to null and confidence to 0 for that store. "description" should be a concise (max ~20 words) English description of the matched product (or empty string if no match).

Respond with ONLY a JSON object of the form {"results": [{"store": "<store name exactly as given above>", "matchIndex": <integer or null>, "confidence": <number 0-1>, "description": "<string>"}, ...]}, with exactly one entry per store listed above. Do not include any other text.`;

  const parsed = await callGroqJson(prompt);
  const list = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.results) ? parsed.results : [];
  const out: Record<string, { matchIndex: number | null; confidence: number; description: string }> = {};
  for (const v of list) {
    if (!v || typeof v !== 'object' || !v.store) continue;
    out[v.store] = {
      matchIndex: v.matchIndex === null || v.matchIndex === undefined ? null : Number(v.matchIndex),
      confidence: Number(v.confidence) || 0,
      description: String(v.description || ''),
    };
  }
  return out;
}

// ── DuckDuckGo search (used for the fallback step) ──────────────────────────
async function ddgSearch(query: string): Promise<Array<{ title: string; url: string; snippet: string }>> {
  const res = await fetchWithTimeout(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`);
  if (!res.ok) return [];
  const html = await res.text();
  const titles = [...html.matchAll(/<a rel="nofollow" class="result__a" href="([^"]+)">([\s\S]*?)<\/a>/g)];
  const snippets = [...html.matchAll(/<a class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g)];
  const out: Array<{ title: string; url: string; snippet: string }> = [];
  for (let i = 0; i < titles.length && out.length < CANDIDATE_LIMIT; i++) {
    let url = titles[i][1];
    try {
      const u = new URL(url.startsWith('//') ? `https:${url}` : url);
      const real = u.searchParams.get('uddg');
      if (real) url = real;
    } catch { /* keep raw url */ }
    out.push({ title: stripHtml(titles[i][2]), url, snippet: stripHtml(snippets[i]?.[1] || '') });
  }
  return out;
}

function hostMatches(url: string, domain: string): boolean {
  try {
    const host = new URL(url).hostname.replace(/^www\./, '');
    const d = domain.replace(/^www\./, '');
    return host === d || host.endsWith(`.${d}`);
  } catch {
    return false;
  }
}

function extractMeta(html: string, prop: string): string {
  const escaped = prop.replace(/[:.]/g, '\\$&');
  let m = html.match(new RegExp(`<meta[^>]+(?:property|name)=["']${escaped}["'][^>]*content=["']([^"']*)["']`, 'i'));
  if (m) return m[1];
  m = html.match(new RegExp(`<meta[^>]+content=["']([^"']*)["'][^>]*(?:property|name)=["']${escaped}["']`, 'i'));
  return m ? m[1] : '';
}

// Best-effort extraction of a product's main image: try Open Graph tags
// first, then fall back to Amazon's inline image attributes/JSON (Amazon
// product pages don't expose og:image to non-browser requests).
function extractProductImage(html: string): string {
  const og = extractMeta(html, 'og:image') || extractMeta(html, 'twitter:image');
  if (og) return og;
  const m = html.match(/data-old-hires="([^"]+)"/) || html.match(/"hiRes":"([^"]+)"/) || html.match(/"large":"([^"]+)"/);
  return m ? m[1].replace(/\\\//g, '/') : '';
}

// Best-effort extraction of image/price/availability from a product page,
// via Open Graph tags and JSON-LD Product/Offer markup.
async function fetchPageMeta(url: string): Promise<{ image: string; price: number | null; inStock: boolean }> {
  try {
    const res = await fetchWithTimeout(url);
    if (!res.ok) return { image: '', price: null, inStock: true };
    const html = await res.text();
    const image = extractProductImage(html);

    let price: number | null = null;
    let availability = '';
    for (const m of html.matchAll(/<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)) {
      try {
        const json = JSON.parse(m[1].trim());
        for (const item of Array.isArray(json) ? json : [json]) {
          const offers = item?.offers ? (Array.isArray(item.offers) ? item.offers[0] : item.offers) : null;
          if (offers?.price != null && price === null) price = parseFloat(String(offers.price));
          if (offers?.availability) availability = String(offers.availability);
        }
      } catch { /* skip invalid JSON-LD block */ }
      if (price !== null && availability) break;
    }
    if (price === null) {
      const metaPrice = extractMeta(html, 'product:price:amount') || extractMeta(html, 'og:price:amount');
      if (metaPrice) price = parsePrice(metaPrice);
    }
    const inStock = !/out\s*of\s*stock|soldout|sold[\s-]?out/i.test(availability);
    return { image, price, inStock };
  } catch {
    return { image: '', price: null, inStock: true };
  }
}

// ── Fallback search via DuckDuckGo, verified by Groq ────────────────────────
// Used for stores with no confident direct match, plus Amazon (which has no
// direct search adapter).
async function searchFallback(
  itemName: string,
  context: string,
  targetStores: string[],
): Promise<Array<{ source: string; buy_url: string; price: number | null; image: string; description: string }>> {
  if (!targetStores.length) return [];

  const candidatesByStore: Record<string, Candidate[]> = {};
  await Promise.allSettled(
    targetStores.map(async (storeKey) => {
      const info = STORE_INFO[storeKey];
      if (!info) return;
      let results = await ddgSearch(`site:${info.domain} ${itemName}`);
      let matches = results.filter((r) => hostMatches(r.url, info.domain));
      if (!matches.length) {
        results = await ddgSearch(`${itemName} ${info.label}`);
        matches = results.filter((r) => hostMatches(r.url, info.domain));
      }
      if (matches.length) {
        candidatesByStore[storeKey] = matches.slice(0, 3).map((r) => ({
          title: r.title,
          description: r.snippet.slice(0, 300),
          url: r.url,
          price: null,
          image: '',
          inStock: true,
        }));
      }
    }),
  );

  if (!Object.keys(candidatesByStore).length) return [];

  const verdicts = await verifyAllMatches(itemName, context, candidatesByStore);

  const results: Array<{ source: string; buy_url: string; price: number | null; image: string; description: string }> = [];
  await Promise.allSettled(
    Object.entries(candidatesByStore).map(async ([storeKey, candidates]) => {
      const v = verdicts[storeKey];
      if (!v || v.matchIndex === null || v.confidence < CONFIDENCE_THRESHOLD) return;
      const c = candidates[v.matchIndex];
      if (!c) return;
      const meta = await fetchPageMeta(c.url);
      if (meta.inStock === false) return;
      results.push({
        source: storeKey,
        buy_url: c.url,
        price: meta.price,
        image: meta.image,
        description: v.description || c.description,
      });
    }),
  );
  return results;
}

// ── main handler ──────────────────────────────────────────────────────────
Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') return new Response(null, { headers: CORS_HEADERS });

  try {
    if (!GROQ_API_KEY) {
      throw new Error('GROQ_API_KEY secret is not configured for this project');
    }

    const { id, table } = await req.json();
    if (!id || !['bom_items', 'inventory'].includes(table)) {
      return new Response(JSON.stringify({ error: "id and table ('bom_items'|'inventory') are required" }), {
        status: 400,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
      });
    }

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
    );

    const { data: row, error: fetchErr } = await supabase.from(table).select('*').eq('id', id).single();
    if (fetchErr || !row) throw fetchErr || new Error('Item not found');

    const context = [row.category, row.notes].filter(Boolean).join(' — ');
    const sources: any[] = Array.isArray(row.sources) ? [...row.sources] : [];

    const found: string[] = [];
    const notFound: string[] = [];
    const errors: Record<string, string> = {};

    // 1. Search every store in parallel; keep only in-stock candidates.
    const candidatesByStore: Record<string, Candidate[]> = {};
    await Promise.allSettled(
      Object.entries(STORE_ADAPTERS).map(async ([storeKey, adapter]) => {
        try {
          const all = await adapter(row.name);
          const inStock = all.filter((c) => c.inStock !== false);
          if (inStock.length) candidatesByStore[storeKey] = inStock;
          else notFound.push(storeKey);
        } catch (err) {
          notFound.push(storeKey);
          errors[storeKey] = err instanceof Error ? err.message : String(err);
        }
      }),
    );

    // 2. One combined Groq call verifies matches across all stores at once.
    if (Object.keys(candidatesByStore).length) {
      try {
        const verdicts = await verifyAllMatches(row.name, context, candidatesByStore);
        await Promise.allSettled(
          Object.entries(candidatesByStore).map(async ([storeKey, candidates]) => {
            const v = verdicts[storeKey];
            if (!v || v.matchIndex === null || v.confidence < CONFIDENCE_THRESHOLD) {
              notFound.push(storeKey);
              return;
            }
            const c = candidates[v.matchIndex];
            if (!c) {
              notFound.push(storeKey);
              return;
            }
            // Backfill image/price from the product page itself when the
            // store's search results didn't include them.
            let image = c.image;
            let price = c.price;
            if (!image || price === null) {
              const meta = await fetchPageMeta(c.url);
              if (!image) image = meta.image;
              if (price === null) price = meta.price;
            }
            const entry = {
              source: storeKey,
              buy_url: c.url,
              price,
              image,
              description: v.description || c.description,
            };
            const idx = sources.findIndex((s) => s.source === storeKey);
            if (idx >= 0) sources[idx] = { ...sources[idx], ...entry };
            else sources.push(entry);
            found.push(storeKey);
          }),
        );
      } catch (err) {
        for (const storeKey of Object.keys(candidatesByStore)) notFound.push(storeKey);
        errors['_verify'] = err instanceof Error ? err.message : String(err);
      }
    }

    // 3. DuckDuckGo + Amazon fallback for everything not yet found, plus Amazon.
    const fallbackTargets = Array.from(new Set([...notFound, 'Amazon']));
    try {
      const fallbackResults = await searchFallback(row.name, context, fallbackTargets);
      for (const r of fallbackResults) {
        const entry = { source: r.source, buy_url: r.buy_url, price: r.price, image: r.image, description: r.description };
        const idx = sources.findIndex((s) => s.source === r.source);
        if (idx >= 0) sources[idx] = { ...sources[idx], ...entry };
        else sources.push(entry);
        found.push(r.source);
        const nfIdx = notFound.indexOf(r.source);
        if (nfIdx >= 0) notFound.splice(nfIdx, 1);
      }
      if (!found.includes('Amazon') && !notFound.includes('Amazon')) notFound.push('Amazon');
    } catch (err) {
      errors['_fallback'] = err instanceof Error ? err.message : String(err);
      if (!found.includes('Amazon') && !notFound.includes('Amazon')) notFound.push('Amazon');
    }

    const { error: updateErr } = await supabase.from(table).update({ sources }).eq('id', id);
    if (updateErr) throw updateErr;

    return new Response(JSON.stringify({ found, notFound, errors }), {
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }), {
      status: 500,
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
    });
  }
});
