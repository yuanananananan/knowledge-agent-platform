import { useQuery } from '@tanstack/react-query';
import { valApi } from '@/services/api/val';
import { valKeys } from './query-keys';

export function useValTasks() {
  return useQuery({
    queryKey: valKeys.lists(),
    queryFn: () => valApi.getValTaskList(),
  });
}