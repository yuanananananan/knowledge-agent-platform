/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_API_TIMEOUT: number;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
