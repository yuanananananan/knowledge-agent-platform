// Model
export type Modality = 'SAR' | 'RD' | 'MON';

export type Dataset = {
  id: number;
  name: string;
  modality: Modality;
  path: string;
};

export type Image = {
  id: number;
  path: string;
  label: string;
  fileName: string;
};
