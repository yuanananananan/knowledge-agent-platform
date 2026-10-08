import axios, {
  type AxiosInstance,
  type AxiosResponse,
  type AxiosError,
  type AxiosRequestConfig,
} from 'axios';
import { apiConfig } from '@/config/env';
import { toast } from 'sonner';

const http: AxiosInstance = axios.create({
  baseURL: apiConfig.baseUrl,
  timeout: apiConfig.timeout,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * 请求拦截器
 * Flask session 认证依赖 cookie，所以每个请求都强制带 withCredentials。
 */
http.interceptors.request.use(
  (config) => {
    config.withCredentials = true;
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

/**
 * 获取错误信息
 */
function getErrorMessage(error: AxiosError): string {
  if (error.response) {
    const { status, data } = error.response;

    const responseData = data as {
      message?: string;
      error?: string;
    };

    switch (status) {
      case 401:
        return responseData?.message || '登录已过期，请重新登录';
      case 403:
        return responseData?.message || '没有权限访问该资源';
      case 404:
        return responseData?.message || '请求的资源不存在';
      case 500:
        return responseData?.message || responseData?.error || '服务器内部错误';
      default:
        return responseData?.message || responseData?.error || `请求失败 (${status})`;
    }
  }

  if (error.request) {
    return '网络连接失败，请检查后端服务是否启动';
  }

  return error.message || '未知错误';
}

/**
 * 响应拦截器
 */
http.interceptors.response.use(
  (response: AxiosResponse) => {
    return response.data;
  },
  (error: AxiosError) => {
    if (!(error.config as (AxiosRequestConfig & { silent?: boolean }) | undefined)?.silent) {
      toast.error(getErrorMessage(error));
    }
    return Promise.reject(error);
  },
);

export const httpService = {
  get<T = unknown>(url: string, config?: AxiosRequestConfig & { silent?: boolean }): Promise<T> {
    return http.get(url, config);
  },

  post<T = unknown>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig,
  ): Promise<T> {
    return http.post(url, data, config);
  },

  put<T = unknown>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig,
  ): Promise<T> {
    return http.put(url, data, config);
  },

  delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return http.delete(url, config);
  },

  patch<T = unknown>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig,
  ): Promise<T> {
    return http.patch(url, data, config);
  },
};

export { http };
export default httpService;
