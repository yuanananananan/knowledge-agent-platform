# 文件用途：多阶段构建并运行 Go Control Plane，最终镜像包含服务程序和数据库迁移文件。
# 构建阶段：下载依赖并编译静态 Control Plane 二进制文件。
FROM golang:1.23-alpine AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -buildvcs=false -trimpath -ldflags="-s -w" -o /control-plane ./cmd/control-plane

# 运行阶段：复制可执行文件与迁移脚本，并以非 root 用户启动。
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /control-plane /control-plane
COPY migrations /migrations
ENV CONTROL_PLANE_MIGRATION=/migrations
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 CMD ["/control-plane", "healthcheck"]
ENTRYPOINT ["/control-plane"]
