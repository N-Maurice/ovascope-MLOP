// Static, descriptive information about the 11 classes. Numbers (counts, metrics) always come from
// the API; only names, colours and short textbook descriptions live here.

export type ClassInfo = { short: string; common: string; group: string; features: string; colour: string };

export const CLASS_INFO: ClassInfo[] = [
  { short: "Ascaris", common: "Large roundworm", group: "Nematode", colour: "#0F766E",
    features: "Round to oval egg with a thick, bumpy (mammillated) outer shell, often stained brown." },
  { short: "Capillaria", common: "Intestinal capillariasis", group: "Nematode", colour: "#B45309",
    features: "Peanut- or barrel-shaped egg with flattened plugs at both ends and a striated shell." },
  { short: "Enterobius", common: "Pinworm", group: "Nematode", colour: "#2563EB",
    features: "Elongated oval egg, flattened on one side, with a thin colourless shell." },
  { short: "Fasciolopsis", common: "Giant intestinal fluke", group: "Trematode", colour: "#7C3AED",
    features: "Very large ellipsoidal egg with a thin shell and a small, indistinct operculum (lid)." },
  { short: "Hookworm", common: "Ancylostoma / Necator", group: "Nematode", colour: "#DB2777",
    features: "Oval egg with a thin, smooth shell; the embryo is often in early cleavage with a clear space around it." },
  { short: "H. diminuta", common: "Rat tapeworm", group: "Cestode", colour: "#0891B2",
    features: "Round, yellowish egg with a thick outer shell; the inner membrane has no polar filaments." },
  { short: "H. nana", common: "Dwarf tapeworm", group: "Cestode", colour: "#65A30D",
    features: "Smaller, oval, colourless egg; the inner membrane has two polar thickenings with filaments." },
  { short: "Opisthorchis", common: "Southeast Asian liver fluke", group: "Trematode", colour: "#CA8A04",
    features: "Small, light-brown operculated egg with 'shoulders' around the lid and a small knob at the other end." },
  { short: "Paragonimus", common: "Lung fluke", group: "Trematode", colour: "#DC2626",
    features: "Large, yellow-brown operculated egg with a thickened shell at the end opposite the lid." },
  { short: "Taenia", common: "Pork / beef tapeworm", group: "Cestode", colour: "#4F46E5",
    features: "Spherical egg with a thick, radially striated brown shell around a six-hooked embryo." },
  { short: "Trichuris", common: "Whipworm", group: "Nematode", colour: "#059669",
    features: "Barrel-shaped egg with a plug at each end and a smooth brown shell." },
];

export const info = (classId: number): ClassInfo =>
  CLASS_INFO[classId] ?? { short: `Class ${classId}`, common: "", group: "", features: "", colour: "#64748B" };

/** Look up a class id from its display name (the API's evaluation files are keyed by name). */
export const idByName = (names: string[], name: string) => names.indexOf(name);
