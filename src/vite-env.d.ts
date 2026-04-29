/// <reference types="vite/client" />

interface Window {
  webkitAudioContext?: typeof AudioContext;
}

type PrototypeXrSessionOptions = {
  optionalFeatures?: string[];
  domOverlay?: {
    root: Element;
  };
};

interface Navigator {
  xr?: {
    isSessionSupported(mode: "immersive-ar"): Promise<boolean>;
    requestSession(mode: "immersive-ar", options?: PrototypeXrSessionOptions): Promise<unknown>;
  };
}
