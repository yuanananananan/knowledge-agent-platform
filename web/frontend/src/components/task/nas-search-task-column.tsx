import type { NasSearchJobData } from '@/types/nas';
import { createColumnHelper } from '@tanstack/react-table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { MoreHorizontal } from 'lucide-react';

const columnHelper = createColumnHelper<NasSearchJobData>();

interface ColumnCallbacks {
  onDelete: (task: NasSearchJobData) => void;
  onViewLogs: (task: NasSearchJobData) => void;
}

export const createNasSearchColumns = ({
  onDelete,
  onViewLogs,
}: ColumnCallbacks) => [
  columnHelper.accessor('jobId', {
    header: '任务ID',
    size: 120,
    cell: (props) => <div className="truncate">{props.getValue()}</div>,
  }),

  columnHelper.accessor('description', {
    header: '描述',
    size: 120,
    cell: (props) => (
      <div className="truncate text-sm text-muted-foreground">
        {props.getValue()}
      </div>
    ),
  }),

  columnHelper.accessor('modality', {
    header: '模态',
    size: 50,
    cell: (props) => <Badge variant="secondary">{props.getValue()}</Badge>,
  }),

  columnHelper.accessor('status', {
    header: '状态',
    size: 60,
    cell: (props) => <Badge>{props.getValue()}</Badge>,
  }),

  columnHelper.accessor('progress', {
    header: '进度',
    size: 90,
    cell: (props) => {
      const value = Number(props.getValue() ?? 0);

      return (
        <div className="space-y-1">
          <Progress value={value} />
          <div className="text-xs text-muted-foreground">{value}%</div>
        </div>
      );
    },
  }),

  columnHelper.accessor('backboneId', {
    header: 'Backbone ID',
    size: 80,
    cell: (props) => {
      const value = props.getValue();

      if (!value) {
        return <span className="text-muted-foreground">-</span>;
      }

      return <Badge variant="secondary">{value}</Badge>;
    },
  }),

  columnHelper.accessor('genotype', {
    header: '最优结构',
    size: 80,
    cell: (props) => {
      const genotype = props.getValue();

      if (!genotype) {
        return <span className="text-muted-foreground">未生成</span>;
      }

      return (
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="link" className="p-0">
              查看
            </Button>
          </DialogTrigger>

          <DialogContent className="max-h-[80vh] max-w-[70vw] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>NAS最优模型结构</DialogTitle>
              <DialogDescription>
                搜索完成后解析得到的 genotype，会注册为 NAS Backbone。
              </DialogDescription>
            </DialogHeader>

            <pre className="overflow-auto rounded-md bg-muted p-4 text-xs">
              {JSON.stringify(genotype, null, 2)}
            </pre>
          </DialogContent>
        </Dialog>
      );
    },
  }),

  columnHelper.accessor('createTime', {
    header: '创建时间',
    size: 100,
    cell: (props) => {
      const time = props.getValue();

      if (!time) {
        return <span className="text-muted-foreground">-</span>;
      }

      return <div className="text-sm">{new Date(time).toLocaleString('zh-CN')}</div>;
    },
  }),

  columnHelper.display({
    id: 'actions',
    header: '操作',
    size: 50,
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
            <DropdownMenuItem onClick={() => onViewLogs(row)}>
              查看日志
            </DropdownMenuItem>

            <DropdownMenuItem
              onClick={() => onDelete(row)}
              className="text-destructive"
            >
              删除任务
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      );
    },
  }),
];