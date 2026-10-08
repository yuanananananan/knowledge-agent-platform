import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@/index.css';
import Layout from '@/layouts/root-layout';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Toaster } from '@/components/ui/sonner';
import LoginPage from './pages/login';
import Dataset from './pages/dataset';
import Feature from './pages/feature';
import User from './pages/user';
import Backbone from './pages/backbone';
import Classifier from './pages/classifier';
import FeatureGroup from './pages/feature-group';
import CreateTask from './pages/create-task';
import Task from './pages/task';
import TaskDetail from './pages/task-detail';
import AgentPage from './pages/agent';

// 创建 QueryClient 实例
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000, // 1分钟内数据保持新鲜
      gcTime: 5 * 60 * 1000, // 5分钟后清除缓存 (以前叫 cacheTime)
      retry: 1, // 失败后重试次数
      refetchOnWindowFocus: false, // 窗口聚焦时不自动重新获取
    },
  },
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/task" replace />} />
            <Route path="dataset" element={<Dataset />} />
            <Route path="classifier" element={<Classifier />} />
            <Route path="backbone" element={<Backbone />} />
            <Route path="feature" element={<Feature />} />
            <Route path="feature_group" element={<FeatureGroup />} />
            <Route path="user" element={<User />} />
            <Route path="create_task" element={<CreateTask />} />
            <Route path="task" element={<Task />} />
            <Route path="task/:job_id" element={<TaskDetail />} />
            <Route path="agent" element={<AgentPage />} />
            <Route path="agent/:conversationId" element={<AgentPage />} />
          </Route>
          <Route path="/login" element={<LoginPage />}></Route>
        </Routes>
        <Toaster />
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </StrictMode>
);
