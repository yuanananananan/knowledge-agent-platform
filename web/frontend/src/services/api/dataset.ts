import { httpService } from '@/services/http';
import type { Dataset, Modality, Image } from '@/types/dataset';

type ImageDTO = {
  dataset_id: number;
  name: string;
  modality: string;
  id: number;
  label: string;
  path: string;
};

export const datasetApi = {
  async getDatasetList() {
    return await httpService.get<Dataset[]>('/api/datasets');
  },
  async uploadDataset(data: { name: string; file: File; modality: Modality }) {
    const formData = new FormData();
    formData.append('name', data.name);
    formData.append('file', data.file);
    formData.append('modality', data.modality);

    return await httpService.post('/api/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  async getImageListByDatasetId(id: number): Promise<Image[]> {
    const data = await httpService.get<ImageDTO[]>(`/api/images/${id}`);
    return data.map((item, index) => {
      const fileName = item.path.split('\\').pop() || `${item.label}_${index}`;
      return {
        id: item.id,
        label: item.label,
        fileName,
        path: item.path,
      };
    });
  },
  getImageUrl(imagePath: string): string {
    const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000';
    const encodedPath = encodeURIComponent(imagePath);
    return `${baseURL}/api/images/file?image_path=${encodedPath}`;
  },
};
