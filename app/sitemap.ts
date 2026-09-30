import type { MetadataRoute } from 'next'
import { SITE } from '@/lib/site'

// Bump lastModified by hand when the page's content meaningfully changes (search engines distrust a date that
// changes on every build). Google ignores changeFrequency and priority, so they are left out.
export default function sitemap(): MetadataRoute.Sitemap {
  return [{ url: `${SITE}/`, lastModified: new Date('2026-09-29') }]
}
