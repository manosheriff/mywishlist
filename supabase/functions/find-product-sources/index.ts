// Supabase Edge Function: find-product-sources
//
// POST { id: number, table: 'bom_items' | 'inventory' }
//
// For the given row, searches each of the 7 online stores for the item's
// name, asks Gemini to verify whether any search result is the same product,
// and on a confident match upserts {source, buy_url, price, image,
// description} into the row's `sources` jsonb array.

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

const GEMINI_API_KEY = Deno.env.get('GEMINI_API_KEY') ?? '';
const GEMINI_MODEL = Deno.env.get('GEMINI_MODEL') || 'gemini-2.5-flash';
const CONFIDENCE_THRESHOLD = 0.6;
const CANDIDATE_LIMIT = 5;
const SEARCH_TIMEOUT_MS = 10000;
const GEMINI_TIMEOUT_MS = 20000;

interface Candidate {
  title: string;
  price: number | null;
  url: string;
  image: string;
  description: string;
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
// known patterns and use whichever matches.
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
    const href = titleM[1].split('?')[0];
    out.push({
      title: titleM[2].trim(),
      url: href.startsWith('http') ? href : `${base}${href}`,
      image: imgM ? normalizeUrl(base, imgM[1]) : '',
      price: priceM ? parsePrice(priceM[1]) : null,
      description: '',
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
    const imgM = inner.match(/<img[^>]*src="([^"]+)"/);
    out.push({
      title: titleM[1].trim(),
      url: `${base}${href}`,
      image: imgM ? normalizeUrl(base, imgM[1]) : '',
      price: priceM ? parsePrice(priceM[1]) : null,
      description: '',
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
      };
    });
  };
}

// ── Odoo adapter (RAM Electronics) ──────────────────────────────────────────
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

// ── Gemini verification ──────────────────────────────────────────────────────
async function verifyMatch(
  itemName: string,
  context: string,
  candidates: Candidate[],
): Promise<{ matchIndex: number | null; confidence: number; description: string }> {
  const list = candidates
    .map((c, i) => `${i}. "${c.title}"${c.description ? ` — ${c.description.slice(0, 200)}` : ''}`)
    .join('\n');

  const prompt = `You are verifying whether any of these store search results are the SAME product as a part needed for a hobby electronics / 3D-printing project.

Item needed: "${itemName}"${context ? `\nContext: ${context}` : ''}

Store search results:
${list}

Pick the result that is the SAME product (same component, value and type), not just similar or related. If none of them are clearly the same product, set matchIndex to null. "description" should be a concise (max ~20 words) English description of the matched product (or empty string if no match). Respond with JSON only.`;

  const body = {
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: 'OBJECT',
        properties: {
          matchIndex: { type: 'INTEGER', nullable: true },
          confidence: { type: 'NUMBER' },
          description: { type: 'STRING' },
        },
        required: ['matchIndex', 'confidence', 'description'],
      },
    },
  };

  const res = await fetchWithTimeout(
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    GEMINI_TIMEOUT_MS,
  );
  if (!res.ok) throw new Error(`Gemini API error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error('Gemini returned no content');
  const parsed = JSON.parse(text);
  return {
    matchIndex: parsed.matchIndex === null || parsed.matchIndex === undefined ? null : Number(parsed.matchIndex),
    confidence: Number(parsed.confidence) || 0,
    description: String(parsed.description || ''),
  };
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

    await Promise.allSettled(
      Object.entries(STORE_ADAPTERS).map(async ([storeKey, adapter]) => {
        try {
          const candidates = await adapter(row.name);
          if (!candidates.length) {
            notFound.push(storeKey);
            return;
          }

          const verdict = await verifyMatch(row.name, context, candidates);
          if (verdict.matchIndex === null || verdict.confidence < CONFIDENCE_THRESHOLD) {
            notFound.push(storeKey);
            return;
          }
          const c = candidates[verdict.matchIndex];
          if (!c) {
            notFound.push(storeKey);
            return;
          }

          const entry = {
            source: storeKey,
            buy_url: c.url,
            price: c.price,
            image: c.image,
            description: verdict.description || c.description,
          };
          const idx = sources.findIndex((s) => s.source === storeKey);
          if (idx >= 0) sources[idx] = { ...sources[idx], ...entry };
          else sources.push(entry);
          found.push(storeKey);
        } catch (err) {
          notFound.push(storeKey);
          errors[storeKey] = err instanceof Error ? err.message : String(err);
        }
      }),
    );

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
