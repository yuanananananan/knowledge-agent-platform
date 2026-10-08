import {
  ImagePlus,
  Database,
  User,
  Users,
  BringToFront,
  Boxes,
  ChevronRight,
  CircleCheckBig,
  Bot,
} from 'lucide-react';

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
  SidebarMenuSub,
  SidebarMenuSubItem,
} from '@/components/ui/sidebar';
import { Button } from './ui/button';
import { useAuthStore } from '@/stores/auth';
import { authApi } from '@/services/api/auth';
import { toast } from 'sonner';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { NavLink } from 'react-router';

// Menu items.
const items = [
  {
    title: '实验 Agent',
    url: '/agent',
    icon: Bot,
    roles: ['admin', 'user'],
  },
  {
    title: '任务管理',
    icon: CircleCheckBig,
    subMenu: [
      { title: '任务列表', url: '/task' },
      { title: '创建任务', url: '/create_task' },
    ],
    roles: ['admin', 'user'],
  },
  {
    title: '数据集管理',
    url: '/dataset',
    icon: Database,
    roles: ['admin', 'user'],
  },
  {
    title: '特征管理',
    icon: ImagePlus,
    subMenu: [
      { title: '特征算子', url: '/feature' },
      { title: '特征组合', url: '/feature_group' },
    ],
    roles: ['admin', 'user'],
  },
  {
    title: '骨干网络',
    url: '/backbone',
    icon: Boxes,
    roles: ['admin', 'user'],
  },
  {
    title: '分类器',
    url: '/classifier',
    icon: BringToFront,
    roles: ['admin', 'user'],
  },
  {
    title: '用户管理',
    url: '/user',
    icon: Users,
    roles: ['admin', 'user'],
  },
];

export function AppSidebar() {
  const { user, isAuthenticated, resetAuthState } = useAuthStore();

  // 过滤有权限的菜单项
  const accessibleItems = items.filter((item) => {
    if (!isAuthenticated || !user) return false;
    return item.roles.includes(user.userRole);
  });

  return (
    <Sidebar>
      {isAuthenticated && user && (
        <SidebarHeader className="p-4">
          <div className="flex items-center gap-2">
            <div className="flex flex-col justify-center flex-1">
              <p className="font-medium">{user.userName}</p>
              <p className="text-sm text-muted-foreground">{user.userAccount}</p>
            </div>
          </div>
        </SidebarHeader>
      )}
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Application</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {accessibleItems.map((item) => {
                if (item.subMenu) {
                  return (
                    <Collapsible key={item.title} defaultOpen className="group/collapsible">
                      <SidebarMenuItem>
                        <CollapsibleTrigger asChild>
                          <SidebarMenuButton>
                            <item.icon className="mr-1 h-4 w-4" />
                            <span>{item.title}</span>
                            <ChevronRight className="ml-auto h-4 w-4 transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                          </SidebarMenuButton>
                        </CollapsibleTrigger>
                        <CollapsibleContent>
                          <SidebarMenuSub>
                            {item.subMenu.map((sub) => (
                              <SidebarMenuSubItem key={sub.title}>
                                <NavLink to={sub.url}>
                                  {({ isActive }) => (
                                    <SidebarMenuButton isActive={isActive}>
                                      <span>{sub.title}</span>
                                    </SidebarMenuButton>
                                  )}
                                </NavLink>
                              </SidebarMenuSubItem>
                            ))}
                          </SidebarMenuSub>
                        </CollapsibleContent>
                      </SidebarMenuItem>
                    </Collapsible>
                  );
                } else {
                  return (
                    <SidebarMenuItem key={item.title}>
                      <NavLink to={item.url}>
                        {({ isActive }) => (
                          <SidebarMenuButton isActive={isActive}>
                            <item.icon className="mr-1 h-4 w-4" />
                            <span>{item.title}</span>
                          </SidebarMenuButton>
                        )}
                      </NavLink>
                    </SidebarMenuItem>
                  );
                }
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      {isAuthenticated && (
        <SidebarFooter className="border-t">
          <Button
            variant="link"
            onClick={async () => {
              try {
                authApi.logout();
                resetAuthState();
              } catch {
                toast.error('登出失败，请重试');
              }
            }}
            className="w-full flex items-center justify-center cursor-pointer text-muted-foreground"
          >
            Logout
            <User className="h-4 w-4" />
          </Button>
        </SidebarFooter>
      )}
    </Sidebar>
  );
}
