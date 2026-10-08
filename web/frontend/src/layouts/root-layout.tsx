import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar';
import { AppSidebar } from '@/components/app-sidebar';
import { Outlet } from 'react-router';
import { Suspense } from 'react';
import { ErrorBoundary } from '@/components/error-boundary';
import { LoadingFallback } from '@/components/loading-fallback';
import { ProtectedRoute } from '@/components/guards/ProtectedRoute';

export default function Layout() {
  return (
    <ProtectedRoute>
      <ErrorBoundary>
        <SidebarProvider>
          <AppSidebar />
          <main className="flex flex-col flex-1">
            <header className="sticky top-0 z-50 flex items-center h-15 p-2 border-b bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60">
              <SidebarTrigger />
            </header>
            <div className="flex-1">
              <Suspense fallback={<LoadingFallback />}>
                <Outlet />
              </Suspense>
            </div>
          </main>
        </SidebarProvider>
      </ErrorBoundary>
    </ProtectedRoute>
  );
}
