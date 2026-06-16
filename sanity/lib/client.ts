import { createClient } from 'next-sanity'

export const client = createClient({
  projectId: 'zapzi9fy',
  dataset: 'production',
  apiVersion: '2024-01-01',
  useCdn: true,
})

export async function getWorkCards() {
  return client.fetch(`*[_type == "work"] | order(order asc)`)
}

export async function getTestimonials() {
  return client.fetch(`*[_type == "testimonial"] | order(order asc)`)
}

export async function getSettings() {
  return client.fetch(`*[_type == "settings"][0]`)
}
