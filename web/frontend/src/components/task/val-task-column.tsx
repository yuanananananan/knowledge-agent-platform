import type { ValTaskData } from '@/types/task';
import { createColumnHelper } from '@tanstack/react-table';
import { Badge } from '../ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';
import { MoreHorizontal } from 'lucide-react';
import { Button } from '../ui/button';
import { Checkbox } from '../ui/checkbox';

const columnHelper = createColumnHelper<ValTaskData>();

export const createColumns = (
  onViewDetail: (jobId: string) => void,
  onDelete: (jobId: string) => void
) => [
  columnHelper.display({
    id: 'select',
    header: ({ table }) => (
      <Checkbox
        checked={
          table.getIsAllPageRowsSelected() || (table.getIsSomePageRowsSelected() && 'indeterminate')
        }
        onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
        aria-label="全选"
      />
    ),
    cell: ({ row }) => (
      <Checkbox
        checked={row.getIsSelected()}
        onCheckedChange={(value) => row.toggleSelected(!!value)}
        aria-label="选择行"
      />
    ),
    size: 40,
    enableSorting: false,
    enableHiding: false,
  }),

  columnHelper.accessor('valJobId', {
    header: '任务ID',
    size: 120,
    cell: (props) => <div className="truncate">{props.getValue()}</div>,
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
    size: 60,
    cell: (props) => <Badge>{props.getValue()}</Badge>,
  }),

  columnHelper.accessor('config', {
    header: '测试参数',
    size: 60,
    cell: (props) => (
      <Dialog>
        <DialogTrigger asChild>
          <Button variant="link" className="p-0">
            查看
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>测试参数</DialogTitle>
            <DialogDescription>当前验证任务的结构化配置。</DialogDescription>
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
            <DropdownMenuItem onClick={() => onViewDetail(row.valJobId)}>
              测试详情
            </DropdownMenuItem>

            <DropdownMenuItem
              onClick={() => onDelete(row.valJobId)}
              className="text-red-400 focus:text-red-400"
            >
              删除
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      );
    },
    size: 30,
  }),
];
