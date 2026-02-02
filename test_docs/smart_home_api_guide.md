# SmartHome Pro API 技术文档

## 版本信息
- **版本**: v2.5.0
- **发布日期**: 2026-01-15
- **兼容性**: Python 3.9+, Node.js 18+

---

## 1. 概述

SmartHome Pro 是一个智能家居控制平台，提供统一的 RESTful API 接口来管理各类智能设备。本文档详细介绍了 API 的使用方法、认证机制和设备控制协议。

### 1.1 系统架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   移动应用       │────▶│   API Gateway   │────▶│   设备控制器     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   消息队列       │
                        │   (RabbitMQ)    │
                        └─────────────────┘
```

### 1.2 核心依赖

| 组件 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.100+ | Web 框架 |
| Redis | 7.0+ | 缓存与会话 |
| PostgreSQL | 15+ | 主数据库 |
| RabbitMQ | 3.12+ | 消息队列 |
| Zigbee2MQTT | 1.30+ | Zigbee 设备桥接 |

---

## 2. 认证机制

### 2.1 OAuth 2.0 认证流程

SmartHome Pro 使用 OAuth 2.0 协议进行用户认证。支持以下授权模式：

1. **授权码模式** (Authorization Code) - 推荐用于 Web 应用
2. **客户端凭证模式** (Client Credentials) - 用于服务器间通信
3. **刷新令牌** (Refresh Token) - 用于延长会话

### 2.2 获取访问令牌

```python
import requests

# 请求访问令牌
response = requests.post(
    "https://api.smarthome.pro/oauth/token",
    data={
        "grant_type": "client_credentials",
        "client_id": "YOUR_CLIENT_ID",
        "client_secret": "YOUR_CLIENT_SECRET",
        "scope": "devices:read devices:write"
    }
)

token = response.json()["access_token"]
```

### 2.3 令牌配置

| 参数 | 值 | 说明 |
|------|------|------|
| access_token 有效期 | 3600秒 | 1小时后过期 |
| refresh_token 有效期 | 2592000秒 | 30天后过期 |
| 最大并发会话 | 5 | 超过将踢出最早会话 |

---

## 3. 设备管理 API

### 3.1 获取设备列表

**端点**: `GET /api/v2/devices`

**请求头**:
```
Authorization: Bearer {access_token}
Content-Type: application/json
```

**响应示例**:
```json
{
    "status": "success",
    "data": {
        "devices": [
            {
                "id": "dev_001",
                "name": "客厅灯",
                "type": "light",
                "protocol": "zigbee",
                "status": "online",
                "capabilities": ["on_off", "brightness", "color_temp"]
            },
            {
                "id": "dev_002", 
                "name": "空调",
                "type": "climate",
                "protocol": "wifi",
                "status": "online",
                "capabilities": ["on_off", "temperature", "mode", "fan_speed"]
            }
        ],
        "total": 2
    }
}
```

### 3.2 控制设备

**端点**: `POST /api/v2/devices/{device_id}/command`

**请求体**:
```json
{
    "command": "set_state",
    "params": {
        "power": "on",
        "brightness": 80,
        "color_temp": 4000
    }
}
```

**错误码**:

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| E1001 | 设备离线 | 检查设备电源和网络连接 |
| E1002 | 命令不支持 | 查看设备 capabilities 列表 |
| E1003 | 参数无效 | 检查参数范围和格式 |
| E1004 | 权限不足 | 确认 token 包含 devices:write 权限 |

---

## 4. 场景自动化

### 4.1 创建场景

场景允许用户定义一组设备动作，可以一键触发或按条件自动执行。

**端点**: `POST /api/v2/scenes`

**请求体**:
```json
{
    "name": "回家模式",
    "trigger": {
        "type": "geofence",
        "params": {
            "latitude": 31.2304,
            "longitude": 121.4737,
            "radius": 100
        }
    },
    "actions": [
        {
            "device_id": "dev_001",
            "command": "set_state",
            "params": {"power": "on", "brightness": 100}
        },
        {
            "device_id": "dev_002",
            "command": "set_state",
            "params": {"power": "on", "temperature": 24, "mode": "cool"}
        }
    ],
    "enabled": true
}
```

### 4.2 触发器类型

| 触发器 | 类型标识 | 参数 |
|--------|----------|------|
| 地理围栏 | geofence | latitude, longitude, radius |
| 时间计划 | schedule | cron_expression |
| 设备状态 | device_state | device_id, condition |
| 手动触发 | manual | 无 |

---

## 5. 实时通信

### 5.1 WebSocket 连接

SmartHome Pro 提供 WebSocket 接口用于实时接收设备状态更新。

**连接地址**: `wss://api.smarthome.pro/ws/v2/events`

**连接示例**:
```javascript
const ws = new WebSocket('wss://api.smarthome.pro/ws/v2/events', {
    headers: {
        'Authorization': `Bearer ${accessToken}`
    }
});

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('设备状态更新:', data);
};
```

### 5.2 事件类型

| 事件类型 | 说明 |
|----------|------|
| device.state_changed | 设备状态变化 |
| device.online | 设备上线 |
| device.offline | 设备离线 |
| scene.triggered | 场景被触发 |
| alert.triggered | 告警触发 |

---

## 6. 故障排除

### 6.1 常见问题

**问题**: 设备无法控制，返回 E1001 错误

**解决方案**:
1. 检查设备电源是否正常
2. 确认设备与网关的距离在有效范围内（Zigbee: 10米，WiFi: 取决于路由器）
3. 重启 Zigbee2MQTT 服务：`sudo systemctl restart zigbee2mqtt`
4. 查看设备日志：`journalctl -u smarthome-controller -f`

**问题**: API 响应缓慢（>2秒）

**解决方案**:
1. 检查 Redis 缓存服务状态
2. 确认 PostgreSQL 连接池配置：建议 `max_connections=100`
3. 启用 API 响应压缩：设置 `ENABLE_GZIP=true`

### 6.2 日志级别配置

在 `config.yaml` 中设置日志级别：

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  format: json
  output:
    - stdout
    - file:/var/log/smarthome/api.log
```

---

## 7. SDK 使用指南

### 7.1 Python SDK

**安装**:
```bash
pip install smarthome-pro-sdk>=2.5.0
```

**使用示例**:
```python
from smarthome_sdk import SmartHomeClient

# 初始化客户端
client = SmartHomeClient(
    api_key="YOUR_API_KEY",
    endpoint="https://api.smarthome.pro"
)

# 获取所有设备
devices = client.devices.list()

# 控制设备
client.devices.control(
    device_id="dev_001",
    command="set_state",
    params={"power": "on", "brightness": 80}
)

# 创建场景
scene = client.scenes.create(
    name="晚安模式",
    actions=[
        {"device_id": "dev_001", "params": {"power": "off"}},
        {"device_id": "dev_002", "params": {"power": "off"}}
    ]
)
```

### 7.2 Node.js SDK

**安装**:
```bash
npm install @smarthome-pro/sdk@^2.5.0
```

**使用示例**:
```javascript
const { SmartHomeClient } = require('@smarthome-pro/sdk');

const client = new SmartHomeClient({
    apiKey: 'YOUR_API_KEY',
    endpoint: 'https://api.smarthome.pro'
});

// 获取设备列表
const devices = await client.devices.list();

// 订阅设备事件
client.events.subscribe('device.state_changed', (event) => {
    console.log(`设备 ${event.device_id} 状态变化:`, event.state);
});
```

---

## 附录 A: 设备协议对照表

| 协议 | 支持设备类型 | 最大设备数 | 延迟 |
|------|-------------|-----------|------|
| Zigbee 3.0 | 灯光、传感器、开关 | 200 | <100ms |
| Z-Wave | 门锁、窗帘 | 232 | <200ms |
| WiFi | 空调、电视、摄像头 | 无限制 | <500ms |
| Bluetooth Mesh | 灯光、传感器 | 32767 | <150ms |
| Matter | 全品类 | 无限制 | <100ms |

## 附录 B: API 速率限制

| 端点类型 | 限制 | 窗口 |
|----------|------|------|
| 认证接口 | 10次 | 1分钟 |
| 设备读取 | 100次 | 1分钟 |
| 设备控制 | 30次 | 1分钟 |
| 场景操作 | 20次 | 1分钟 |

超过限制将返回 HTTP 429 状态码。

---

**联系支持**: support@smarthome.pro  
**开发者社区**: https://developers.smarthome.pro  
**GitHub**: https://github.com/smarthome-pro/api-sdk
