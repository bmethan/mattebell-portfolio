import { defineType, defineField } from 'sanity'

export const settingsSchema = defineType({
  name: 'settings',
  title: 'Site Settings',
  type: 'document',
  fields: [
    defineField({ name: 'bio', title: 'Bio (paragraph 1)', type: 'text' }),
    defineField({ name: 'bio2', title: 'Bio (paragraph 2)', type: 'text' }),
    defineField({ name: 'bio3', title: 'Bio (paragraph 3)', type: 'text' }),
    defineField({ name: 'bio4', title: 'Bio (paragraph 4)', type: 'text' }),
    defineField({ name: 'available', title: 'Available for hire?', type: 'boolean' }),
    defineField({ name: 'reelUrl', title: 'Showreel URL (Vimeo embed)', type: 'url' }),
    defineField({ name: 'resumeUrl', title: 'Resume download URL', type: 'url' }),
  ],
})
