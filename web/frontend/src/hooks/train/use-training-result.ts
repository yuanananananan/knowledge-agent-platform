import { useQuery } from '@tanstack/react-query';
import { trainApi } from '@/services/api/train';
import { trainKeys } from './query-keys';

export function useTrainingResult(jobId: string) {
  return useQuery({
    queryKey: trainKeys.result(jobId),
    queryFn: () => trainApi.getTrainingResult(jobId),
    enabled: !!jobId,
  });
}
