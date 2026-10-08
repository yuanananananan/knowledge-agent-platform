export type Backbone = {
  id: number;
  name: string;
  modality?: string;
  backboneType?: string;
  paramsSchema?: Record<string, unknown>;
  artifactPath?: string | null;
  description: string;
};