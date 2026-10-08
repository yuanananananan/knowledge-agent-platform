import { httpService } from '@/services/http';
import type { Classifier } from '@/types/classifier';

export const classifierApi = {
  async getClassifierList(): Promise<Classifier[]> {
    return await httpService.get<Classifier[]>('/api/classifier/');
  },
};
