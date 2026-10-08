import { useQuery } from '@tanstack/react-query';
import { valApi } from '@/services/api/val';
import { valKeys } from './query-keys';

export function useValidationResult(jobId: string) {
  return useQuery({
    queryKey: valKeys.result(jobId),
    queryFn: () => valApi.getValidationResult(jobId),
    enabled: !!jobId,
  });
}
