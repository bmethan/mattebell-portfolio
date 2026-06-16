import { defineType, defineField } from 'sanity'

export const workSchema = defineType({
  name: 'work',
  title: 'Work',
  type: 'document',
  fields: [
    defineField({ name: 'title', title: 'Title', type: 'string' }),
    defineField({ name: 'year', title: 'Year', type: 'string' }),
    defineField({ name: 'type', title: 'Type (e.g. Feature film)', type: 'string' }),
    defineField({ name: 'badge', title: 'Badge (e.g. Oscar-nominated production)', type: 'string' }),
    defineField({ name: 'role', title: 'Role', type: 'string' }),
    defineField({ name: 'studio', title: 'Studio', type: 'string' }),
    defineField({ name: 'icon', title: 'Icon (movie / device-tv / speakerphone)', type: 'string' }),
    defineField({ name: 'order', title: 'Display order', type: 'number' }),
  ],
  orderings: [{ title: 'Display order', name: 'orderAsc', by: [{ field: 'order', direction: 'asc' }] }],
})
