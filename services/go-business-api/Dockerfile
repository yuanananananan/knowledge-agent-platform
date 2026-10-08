# 文件用途：多阶段构建并运行 Go Business API，最终镜像只保留可执行文件和运行时证书。
# 构建阶段：下载依赖并生成静态链接、裁剪调试符号的二进制文件。
FROM golang:1.23-alpine AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -buildvcs=false -trimpath -ldflags="-s -w" -o /business-api ./cmd/business-api

# 运行阶段：使用无 Shell、非 root 的 distroless 镜像缩小攻击面。
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /business-api /business-api
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 CMD ["/business-api", "healthcheck"]
ENTRYPOINT ["/business-api"]
