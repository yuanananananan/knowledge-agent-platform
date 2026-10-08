import { useQuery } from '@tanstack/react-query';
import { datasetKeys } from './query-keys';
import { datasetApi } from '@/services/api/dataset';

export function useImages(datasetId: number, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: datasetKeys.images(datasetId),
    queryFn: () => datasetApi.getImageListByDatasetId(datasetId),
    enabled: options?.enabled ?? datasetId > 0,
  });
}
