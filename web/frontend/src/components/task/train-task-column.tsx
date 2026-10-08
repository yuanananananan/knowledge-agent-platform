import type { TrainTaskData } from '@/types/task';
import { createColumnHelper } from '@tanstack/react-table';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';
import { MoreHorizontal } from 'lucide-react';
import { Button } from '../ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../ui/dialog';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';

const columnHelper = createColumnHelper<TrainTaskData>();

interface ColumnCallbacks {
  onViewDetail: (jobId: string) => void;
  onStartVal: (jobId: string) => void;
  onDelete: (task: TrainTaskData) => void;
}

export const createColumns = ({ onViewDetail, onStartVal, onDelete }: ColumnCallbacks) => [
  columnHelper.accessor('jobId', {
    header: '任务ID',
    size: 120,
    cell: (props) => <div className="truncate">{props.getValue()}</div>,
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
  columnHelper.accessor('createTime', {
    header: '创建时间',
    size: 60,
    cell: (props) => {
      const time = props.getValue();
      const date = new Date(time);
      return <div className="text-sm">{date.toLocaleString('zh-CN')}</div>;
    },
  }),
  columnHelper.accessor('status', {
    header: '状态',
    size: 40,
    cell: (props) => <Badge>{props.getValue()}</Badge>,
  }),
  columnHelper.accessor('config', {
    header: '训练参数',
    size: 30,
    cell: (props) => (
      <Dialog>
        <DialogTrigger asChild>
          <Button variant="link" className="p-0">
            查看
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>训练参数</DialogTitle>
            <DialogDescription>当前训练任务的结构化配置。</DialogDescription>
          </DialogHeader>
          <pre className="text-xs overflow-auto">
            {JSON.stringify(props.getValue(), null, 2)}
          </pre>
        </DialogContent>
      </Dialog>
    ),
  }),
  columnHelper.display({
    id: 'actions',
    header: '操作',
    cell: (props) => {
      const row = props.row.original;
      return (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <span className="sr-only">打开菜单</span>
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onViewDetail(row.jobId)}>训练详情</DropdownMenuItem>
            <DropdownMenuItem onClick={() => onStartVal(row.jobId)}>启动测试</DropdownMenuItem>
            <DropdownMenuItem onClick={() => onDelete(row)} className="text-destructive">
              删除任务
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      );
    },
    size: 30,
  }),
];
