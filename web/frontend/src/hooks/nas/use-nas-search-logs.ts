import { useQuery } from '@tanstack/react-query';
import { nasApi } from '@/services/api/nas';
import { nasKeys } from './query-keys';

export function useNasSearchLogs(jobId?: string) {
  return useQuery({
    queryKey: jobId ? nasKeys.logs(jobId) : nasKeys.logs(''),
    queryFn: () => nasApi.getNasSearchLogs(jobId!),
    enabled: !!jobId,
  });
}