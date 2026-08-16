import { requestUrl } from "obsidian";
import { ExtractResponse } from "./types";

/** Cache keyed by note path. Lives for the lifetime of the view. */
export class ExtractCache {
  private map = new Map<string, ExtractResponse>();

  get(path: string) {
    return this.map.get(path);
  }
  set(path: string, value: ExtractResponse) {
    this.map.set(path, value);
  }
  has(path: string) {
    return this.map.has(path);
  }
  delete(path: string) {
    this.map.delete(path);
  }
  clear() {
    this.map.clear();
  }
}

export async function extract(serverUrl: string, text: string): Promise<ExtractResponse> {
  const resp = await requestUrl({
    url: serverUrl,
    method: "POST",
    contentType: "application/json",
    body: JSON.stringify({ text }),
    throw: false,
  });
  if (resp.status < 200 || resp.status >= 300) {
    throw new Error(`HTTP ${resp.status}${resp.text ? `: ${resp.text.slice(0, 200)}` : ""}`);
  }
  return resp.json as ExtractResponse;
}
