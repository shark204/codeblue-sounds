'use client';

/**
 * SSR-safe wrapper around browser localStorage for the 3D Anatomy Atlas.
 */

import { logEvent } from "./appLog";

type Listener = () => void;

let failure: string | null = null;
const listeners = new Set<Listener>();

export function storageFailure(): string | null {
  return failure;
}

export function onStorageFailure(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function note(error: unknown): void {
  if (typeof window === "undefined") return;
  if (failure !== null) return;
  failure =
    error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  console.error("[storage] localStorage is unavailable:", error);
  logEvent("error", "storage", failure);
  for (const listener of listeners) listener();
}

export function readLocal(key: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(key);
  } catch (error) {
    note(error);
    return null;
  }
}

export function writeLocal(key: string, value: string): boolean {
  if (typeof window === "undefined") return false;
  try {
    window.localStorage.setItem(key, value);
    return true;
  } catch (error) {
    note(error);
    return false;
  }
}

export function removeLocal(key: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(key);
  } catch (error) {
    note(error);
  }
}

const STAMP = "anatria3d.storage.stamp";

export function checkStorage(): void {
  if (typeof window === "undefined") return;
  const key = "anatria3d.storage.probe";
  if (!writeLocal(key, "1")) return;
  removeLocal(key);
}

export function confirmStoragePersists(hasHistory: boolean): void {
  if (typeof window === "undefined") return;
  const previous = readLocal(STAMP);
  if (previous === null && hasHistory && !anythingStored()) {
    note(
      new Error(
        "nothing kept by earlier launches survived — the browser engine is " +
          "storing settings in memory only",
      ),
    );
  }
  writeLocal(STAMP, String(Date.now()));
}

function anythingStored(): boolean {
  if (typeof window === "undefined") return false;
  try {
    for (let index = 0; index < window.localStorage.length; index += 1) {
      const key = window.localStorage.key(index);
      if (key !== null && key !== STAMP && key.startsWith("anatria3d.")) {
        return true;
      }
    }
    return false;
  } catch (error) {
    note(error);
    return false;
  }
}
