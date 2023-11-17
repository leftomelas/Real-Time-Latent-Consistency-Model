import { writable } from 'svelte/store';
import { PUBLIC_WSS_URL } from '$env/static/public';


export enum LCMLiveStatus {
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
    WAIT = "wait",
}

const initStatus: LCMLiveStatus = LCMLiveStatus.DISCONNECTED;

export const lcmLiveStatus = writable<LCMLiveStatus>(initStatus);
export const streamId = writable<string | null>(null);

let websocket: WebSocket | null = null;
export const lcmLiveActions = {
    async start() {
        return new Promise((resolve, reject) => {

            try {
                const websocketURL = PUBLIC_WSS_URL ? PUBLIC_WSS_URL : `${window.location.protocol === "https:" ? "wss" : "ws"
                    }:${window.location.host}/ws`;

                websocket = new WebSocket(websocketURL);
                websocket.onopen = () => {
                    console.log("Connected to websocket");
                };
                websocket.onclose = () => {
                    lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                    console.log("Disconnected from websocket");
                };
                websocket.onerror = (err) => {
                    console.error(err);
                };
                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    console.log("WS: ", data);
                    switch (data.status) {
                        case "connected":
                            const userId = data.userId;
                            lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                            streamId.set(userId);
                            break;
                        case "timeout":
                            console.log("timeout");
                            lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                            streamId.set(null);
                            reject("timeout");
                        case "error":
                            console.log(data.message);
                            lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                            streamId.set(null);
                            reject(data.message);
                    }
                };

            } catch (err) {
                console.error(err);
                lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                streamId.set(null);

                reject(err);
            }
        });
    },
    send(data: Blob | { [key: string]: any }) {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (data instanceof Blob) {
                websocket.send(data);
            } else {
                websocket.send(JSON.stringify(data));
            }
        } else {
            console.log("WebSocket not connected");
        }
    },
    async stop() {

        if (websocket) {
            websocket.close();
        }
        websocket = null;
        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
        streamId.set(null);
    },
};