import { createColumnHelper } from '@tanstack/react-table';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Button } from '@/components/ui/button';
import type { FeatureGroupWithDetails } from '@/types/feature';

const columnHelper = createColumnHelper<FeatureGroupWithDetails>();

interface ColumnCallbacks {
  onDetailsClick: (group: FeatureGroupWithDetails) => void;
}

export const createColumns = (callbacks: ColumnCallbacks) => [
  columnHelper.accessor('id', {
    header: 'ID',
    size: 60,
  }),
  columnHelper.accessor('name', {
    header: '组合名称',
    size: 100,
  }),
  columnHelper.accessor('description', {
    header: '描述',
    cell: (props) => {
      const description = props.getValue();
      return (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="text-sm text-muted-foreground truncate cursor-help">
                {description}
              </div>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-md">
              <p>{description}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    },
    size: 120,
  }),
  columnHelper.accessor('featureCount', {
    header: '特征数量',
    size: 60,
  }),
  columnHelper.display({
    id: 'details',
    header: '组合详情',
    cell: (props) => {
      const group = props.row.original;
      return (
        <Button
          variant="link"
          onClick={() => {
            callbacks.onDetailsClick(group);
          }}
        >
          查看
        </Button>
      );
    },
    size: 60,
  }),
  columnHelper.accessor('createTime', {
    header: '创建时间',
    cell: (props) => {
      const time = props.getValue();
      const date = new Date(time);
      return <div className="text-sm">{date.toLocaleString('zh-CN')}</div>;
    },
    size: 100,
  }),
];
