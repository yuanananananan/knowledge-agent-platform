import { useQuery } from '@tanstack/react-query';
import { trainApi } from '@/services/api/train';
import { trainKeys } from './query-keys';

export function useTrainingLogs(jobId: string) {
  return useQuery({
    queryKey: trainKeys.logs(jobId),
    queryFn: () => trainApi.getTrainingLogs(jobId),
    enabled: !!jobId,
  });
}
