import { defineType, defineField } from 'sanity'

export const testimonialSchema = defineType({
  name: 'testimonial',
  title: 'Testimonial',
  type: 'document',
  fields: [
    defineField({ name: 'quote', title: 'Quote', type: 'text' }),
    defineField({ name: 'name', title: 'Name', type: 'string' }),
    defineField({ name: 'title', title: 'Title / Role', type: 'string' }),
    defineField({ name: 'initials', title: 'Initials (2 letters)', type: 'string' }),
    defineField({ name: 'order', title: 'Display order', type: 'number' }),
  ],
})
