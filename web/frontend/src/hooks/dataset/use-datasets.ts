import { useQuery } from '@tanstack/react-query';
import { datasetKeys } from './query-keys';
import { datasetApi } from '@/services/api/dataset';

export function useDatasets() {
  return useQuery({
    queryKey: datasetKeys.list(),
    queryFn: datasetApi.getDatasetList,
    select: (data) => {
      return data;
    },
  });
}
