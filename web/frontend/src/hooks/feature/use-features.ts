import { useQuery } from '@tanstack/react-query';
import { featureKeys } from './query-keys';
import { featureApi } from '@/services/api/feature';

export function useFeatures() {
  return useQuery({
    queryKey: featureKeys.list(),
    queryFn: featureApi.getFeatureList,
    select: (data) => {
      return data;
    },
  });
}
