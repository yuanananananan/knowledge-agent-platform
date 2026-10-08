import { useQuery } from '@tanstack/react-query';
import { nasApi } from '@/services/api/nas';
import { nasKeys } from './query-keys';

export function useNasSearchJobs() {
  return useQuery({
    queryKey: nasKeys.lists(),
    queryFn: nasApi.getNasSearchJobList,
    refetchInterval: 3000,
  });
}