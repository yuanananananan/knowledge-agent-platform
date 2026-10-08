import { useQuery } from '@tanstack/react-query';
import { valApi } from '@/services/api/val';
import { valKeys } from './query-keys';

export function useValidationLogs(jobId: string) {
  return useQuery({
    queryKey: valKeys.logs(jobId),
    queryFn: () => valApi.getValidationLogs(jobId),
    enabled: !!jobId,
  });
}
