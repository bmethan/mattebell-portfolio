export function getYearsExperience(): number {
  const START_YEAR = 1998
  const now = new Date()
  const anniversary = new Date(now.getFullYear(), 5, 8) // June 8 (month is 0-indexed)
  const currentYear = now >= anniversary ? now.getFullYear() : now.getFullYear() - 1
  return currentYear - START_YEAR
}
