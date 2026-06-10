// Supabase Edge Function: find-product-sources
//
// POST { id: number, table: 'bom_items' | 'inventory' }
//
// For the given row, searches each of the 7 online stores for the item's
// name, then makes ONE combined Gemini call to verify which (if any) search
// result from each store is the same product. Stores with no confident match
// (plus Amazon, which has no direct search adapter) are then looked up via a
// single Gemini call using Google Search grounding. On a confident, in-stock
// match, upserts {source, buy_url, price, image, description} into the row's
// `sources` jsonb array.

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

const GEMINI_API_KEY = Deno.env.get('GEMINI_API_KEY') ?? '';
const GEMINI_MODEL = Deno.env.get('GEMINI_MODEL') || 'gemini-2.5-flash';
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;
const CONFIDENCE_THRESHOLD = 0.6;
const CANDIDATE_LIMIT = 5;
const SEARCH_TIMEOUT_MS = 10000;
const GEMINI_TIMEOUT_MS = 25000;
const GEMINI_GROUNDING_TIMEOUT_MS = 30000;

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
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; BOMPortalBot/1.0)', ...(opts.headers || {}) },
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
// Different themes render search results differently, so try a couple of
// known patterns and use whichever matches. "Sold out" detection is
// best-effort: it relies on the price slot showing "Sold out" or a
// badge--sold-out class near the product card.
function parseShopifyGridTheme(html: string, base: string): Candidate[] {
  const marker = '<div class="product grid__item';
  const idxs: number[] = [];
  let pos = 0;
  while (true) {
    const i = html.indexOf(marker, pos);
    if (i === -1) break;
    idxs.push(i);
    pos = i + marker.length;
  }
  const out: Candidate[] = [];
  for (let k = 0; k < idxs.length; k++) {
    const chunk = html.slice(idxs[k], idxs[k + 1] ?? idxs[k] + 4000);
    const titleM = chunk.match(/product__title[^>]*>\s*<a href="([^"]+)">([^<]+)<\/a>/);
    if (!titleM) continue;
    const imgM = chunk.match(/data-src="([^"]+)"/) || chunk.match(/<img[^>]*src="([^"]+)"/);
    const priceM = chunk.match(/product__price">\s*(?:<span class="visually-hidden">[^<]*<\/span>\s*)?([^<]+)/);
    const priceText = priceM ? priceM[1] : '';
    const soldOut = /sold[\s-]?out/i.test(priceText) || /badge--sold-out/i.test(chunk.slice(0, 1200));
    const href = titleM[1].split('?')[0];
    out.push({
      title: titleM[2].trim(),
      url: href.startsWith('http') ? href : `${base}${href}`,
      image: imgM ? normalizeUrl(base, imgM[1]) : '',
      price: priceM ? parsePrice(priceText) : null,
      description: '',
      inStock: !soldOut,
    });
  }
  return out;
}

function parseShopifySimpleTheme(html: string, base: string): Candidate[] {
  const out: Candidate[] = [];
  const re = /<a href="(\/products\/[^"?]+)[^"]*"\s+id="product-\d+"[^>]*>([\s\S]*?)<\/a>/g;
  let m;
  while ((m = re.exec(html))) {
    const href = m[1];
    const inner = m[2];
    const titleM = inner.match(/<h3>([^<]+)<\/h3>/);
    if (!titleM) continue;
    const priceM = inner.match(/<h4>([^<]+)<\/h4>/);
    const priceText = priceM ? priceM[1] : '';
    const imgM = inner.match(/<img[^>]*src="([^"]+)"/);
    const soldOut = /sold[\s-]?out/i.test(priceText) || /sold[\s-]?out/i.test(inner);
    out.push({
      title: titleM[1].trim(),
      url: `${base}${href}`,
      image: imgM ? normalizeUrl(base, imgM[1]) : '',
      price: priceM ? parsePrice(priceText) : null,
      description: '',
      inStock: !soldOut,
    });
  }
  return out;
}

function shopifyAdapter(base: string) {
  return async (q: string): Promise<Candidate[]> => {
    const res = await fetchWithTimeout(`${base}/search?q=${encodeURIComponent(q)}&type=product`);
    if (!res.ok) return [];
    const html = await res.text();
    let candidates = parseShopifyGridTheme(html, base);
    if (!candidates.length) candidates = parseShopifySimpleTheme(html, base);
    return candidates.slice(0, CANDIDATE_LIMIT);
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
      out.push({
        title,
        price: parsePrice(priceM[1]),
        url: href.startsWith('http') ? href : `${base}${href}`,
        image: imgM ? normalizeUrl(base, imgM[1]) : '',
        description: '',
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

// store key -> domain/label, used for the Google/Amazon fallback prompt
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

// ── Gemini verification: ONE call covering every store's candidates ────────
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

For EACH store listed above, pick the result that is the SAME product (same component, value and type), not just similar or related. If none of a store's results are clearly the same product, set matchIndex to null and confidence to 0 for that store. "description" should be a concise (max ~20 words) English description of the matched product (or empty string if no match). Respond with a JSON array containing exactly one entry per store listed, each with fields: store (the exact store name as given above), matchIndex, confidence, description.`;

  const body = {
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: 'ARRAY',
        items: {
          type: 'OBJECT',
          properties: {
            store: { type: 'STRING' },
            matchIndex: { type: 'INTEGER', nullable: true },
            confidence: { type: 'NUMBER' },
            description: { type: 'STRING' },
          },
          required: ['store', 'matchIndex', 'confidence', 'description'],
        },
      },
    },
  };

  const res = await fetchWithTimeout(
    GEMINI_URL,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    GEMINI_TIMEOUT_MS,
  );
  if (!res.ok) throw new Error(`Gemini API error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error('Gemini returned no content');
  const parsed = JSON.parse(text);
  const out: Record<string, { matchIndex: number | null; confidence: number; description: string }> = {};
  for (const v of parsed) {
    out[v.store] = {
      matchIndex: v.matchIndex === null || v.matchIndex === undefined ? null : Number(v.matchIndex),
      confidence: Number(v.confidence) || 0,
      description: String(v.description || ''),
    };
  }
  return out;
}

// ── Google/Amazon fallback via Gemini Google Search grounding ──────────────
// Used for stores with no confident direct match, plus Amazon (which has no
// direct search adapter). Grounding is incompatible with responseSchema, so
// the response is plain text and JSON is extracted manually.
async function searchGoogleFallback(
  itemName: string,
  context: string,
  targetStores: string[],
): Promise<Array<{ source: string; buy_url: string; price: number | null; image: string; description: string }>> {
  if (!targetStores.length) return [];

  const storeList = targetStores
    .map((s) => `- ${STORE_INFO[s]?.label || s} (${STORE_INFO[s]?.domain || s})`)
    .join('\n');
  const validSources = targetStores.join(', ');

  const prompt = `A user needs to buy a part called "${itemName}"${context ? ` (category/context: ${context})` : ''}. Search Google (including Google Shopping) to find where to buy this, and check specifically whether any of these online stores sell it:
${storeList}

For each store with a confident matching product that appears to be in stock (not sold out / unavailable), output an object with fields: source, buy_url (direct product page URL), price (number in EGP if shown, else null), image (product image URL if known, else empty string), description (short English description, max 20 words). source must be exactly one of: ${validSources}. Output ONLY a raw JSON array, no markdown formatting, no commentary - an empty array if nothing confident is found.`;

  const body = {
    contents: [{ parts: [{ text: prompt }] }],
    tools: [{ google_search: {} }],
  };

  const res = await fetchWithTimeout(
    GEMINI_URL,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    GEMINI_GROUNDING_TIMEOUT_MS,
  );
  if (!res.ok) throw new Error(`Gemini grounding API error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) return [];

  let cleaned = text.trim();
  const fenceMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) cleaned = fenceMatch[1].trim();
  const arrMatch = cleaned.match(/\[[\s\S]*\]/);
  if (!arrMatch) return [];

  let parsed: any[];
  try {
    parsed = JSON.parse(arrMatch[0]);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) return [];

  return parsed
    .filter((r) => r && typeof r === 'object' && targetStores.includes(r.source) && r.buy_url)
    .map((r) => ({
      source: String(r.source),
      buy_url: String(r.buy_url),
      price: r.price === null || r.price === undefined || r.price === '' ? null : (Number(r.price) || null),
      image: r.image ? String(r.image) : '',
      description: r.description ? String(r.description) : '',
    }));
}

// ── main handler ──────────────────────────────────────────────────────────
Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') return new Response(null, { headers: CORS_HEADERS });

  try {
    if (!GEMINI_API_KEY) {
      throw new Error('GEMINI_API_KEY secret is not configured for this project');
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

    // 2. One combined Gemini call verifies matches across all stores at once.
    if (Object.keys(candidatesByStore).length) {
      try {
        const verdicts = await verifyAllMatches(row.name, context, candidatesByStore);
        for (const [storeKey, candidates] of Object.entries(candidatesByStore)) {
          const v = verdicts[storeKey];
          if (!v || v.matchIndex === null || v.confidence < CONFIDENCE_THRESHOLD) {
            notFound.push(storeKey);
            continue;
          }
          const c = candidates[v.matchIndex];
          if (!c) {
            notFound.push(storeKey);
            continue;
          }
          const entry = {
            source: storeKey,
            buy_url: c.url,
            price: c.price,
            image: c.image,
            description: v.description || c.description,
          };
          const idx = sources.findIndex((s) => s.source === storeKey);
          if (idx >= 0) sources[idx] = { ...sources[idx], ...entry };
          else sources.push(entry);
          found.push(storeKey);
        }
      } catch (err) {
        for (const storeKey of Object.keys(candidatesByStore)) notFound.push(storeKey);
        errors['_verify'] = err instanceof Error ? err.message : String(err);
      }
    }

    // 3. Google/Amazon fallback for everything not yet found, plus Amazon.
    const fallbackTargets = Array.from(new Set([...notFound, 'Amazon']));
    try {
      const fallbackResults = await searchGoogleFallback(row.name, context, fallbackTargets);
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
      errors['_google_fallback'] = err instanceof Error ? err.message : String(err);
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
