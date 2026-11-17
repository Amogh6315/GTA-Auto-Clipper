import { constantTimeHexEqual, guideText, hmacHex } from '../_lib/razorpay.mjs';

export async function onRequestGet({ request, env }) {
  const token = (request.headers.get('cookie') || '').split('; ')
    .find((item) => item.startsWith('gta_guide='))?.slice('gta_guide='.length);
  const parts = token?.split('.') || [];
  if (parts.length !== 3 || !/^\d+$/.test(parts[0]) || Number(parts[0]) < Math.floor(Date.now() / 1000)) {
    return new Response('A verified purchase is required to download the guide.', { status: 401 });
  }
  const expected = await hmacHex(env.RAZORPAY_KEY_SECRET || '', `guide-session|${parts[0]}.${parts[1]}`);
  if (!env.RAZORPAY_KEY_SECRET || !constantTimeHexEqual(expected, parts[2])) {
    return new Response('A verified purchase is required to download the guide.', { status: 401 });
  }
  const guide = await guideText(env);
  if (!guide) return new Response('Guide delivery is temporarily unavailable.', { status: 503 });
  return new Response(guide, {
    headers: {
      'content-type': 'text/markdown; charset=utf-8',
      'content-disposition': 'attachment; filename="gta-auto-clipper-guide.md"',
      'cache-control': 'private, no-store',
    },
  });
}
