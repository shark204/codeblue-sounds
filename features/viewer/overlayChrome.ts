/**
 * Warm Dark Stone Overlay Chrome (#242120 surface, #38342f border, #ece7e0 primary text, #a39b90 secondary text).
 */

export const OVERLAY_GROUND = "border border-[#38342f] bg-[#242120]/95 text-[#a39b90]";

export const OVERLAY_CHIP = `rounded-lg ${OVERLAY_GROUND} px-2 py-0.5 font-mono text-[10px]`;

export const OVERLAY_CHIP_ACTION = "hover:border-[#575047] hover:text-[#ece7e0]";

export const OVERLAY_SWITCH_OFF =
  "border-[#38342f] bg-[#242120]/95 text-[#a39b90] hover:border-[#575047] hover:text-[#ece7e0]";

export const OVERLAY_SWITCH_ON = {
  cyan: "border-[#575047] bg-[#2a2624] text-[#ece7e0]",
  sky: "border-[#575047] bg-[#2a2624] text-[#ece7e0]",
  rose: "border-[#d98a96]/60 bg-[#2a2624] text-[#ece7e0]",
} as const;
