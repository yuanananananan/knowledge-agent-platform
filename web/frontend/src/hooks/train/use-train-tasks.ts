import { useQuery } from '@tanstack/react-query';
import { trainApi } from '@/services/api/train';
import { trainKeys } from './query-keys';

export function useTrainTasks() {
  return useQuery({
    queryKey: trainKeys.lists(),
    queryFn: trainApi.getTrainTaskList,
  });
}
