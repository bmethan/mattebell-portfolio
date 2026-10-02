// Blackbody color temperature to the ACEScg working space, normalized to unit luminance.
//
// Planckian locus chromaticity: the cubic spline of B. Kang, O. Moon, C. Hong, H. Lee, B. Cho, Y. Kim,
// "Design of Advanced Color-Temperature Control System for HDTV Applications", J. Korean Physical Society
// 41(6):865-871, 2002, Eq. 8-9 (coefficients as printed there; the related US Patent 7,024,034, often cited as
// "Kim et al.", prints slightly different values). Valid 1667 K to 25000 K. XYZ to linear sRGB uses the
// IEC 61966-2-1 Amd.1:2003 matrix; Rec.709 to ACEScg uses the OpenColorIO 2.5 cg-config matrix (Bradford).

const XYZ_TO_REC709 = [
  [3.2406255, -1.537208, -0.4986286],
  [-0.9689307, 1.8757561, 0.0415175],
  [0.0557101, -0.2040211, 1.0569959],
]
export const REC709_TO_AP1 = [
  [0.6130974, 0.33952314, 0.04737945],
  [0.07019372, 0.9163539, 0.0134524],
  [0.02061559, 0.10956977, 0.86981463],
]
const mul = (m: number[][], v: number[]) => m.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2])

export function kelvinToACEScg(kelvin: number): [number, number, number] {
  const T = Math.min(25000, Math.max(1667, kelvin))
  const u = 1000 / T
  const x =
    T <= 4000
      ? ((-0.2661239 * u - 0.2343589) * u + 0.8776956) * u + 0.17991
      : ((-3.0258469 * u + 2.1070379) * u + 0.2226347) * u + 0.24039
  const y =
    T <= 2222
      ? ((-1.1063814 * x - 1.3481102) * x + 2.18555832) * x - 0.20219683
      : T <= 4000
        ? ((-0.9549476 * x - 1.37418593) * x + 2.09137015) * x - 0.16748867
        : ((3.081758 * x - 5.8733867) * x + 3.75112997) * x - 0.37001483

  const XYZ = [x / y, 1, (1 - x - y) / y]
  // Below about 1900 K the locus leaves the Rec.709 gamut; clip and re-normalize luminance.
  const rec709 = mul(XYZ_TO_REC709, XYZ).map(c => Math.max(0, c))
  const l = 0.2126 * rec709[0] + 0.7152 * rec709[1] + 0.0722 * rec709[2]
  const ap1 = mul(REC709_TO_AP1, rec709.map(c => c / l))
  return [ap1[0], ap1[1], ap1[2]]
}
