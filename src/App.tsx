import React, { useState, useRef, useEffect, useCallback } from 'react';

// --- Types ---
interface BoundingBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
}

interface DetectionResult {
  status: 'Safe' | 'Contaminated' | 'Inconclusive' | 'No Fish Detected';
  confidence: number;
  description: string;
  guidance: string;
  timestamp: string;
  image: string;
  location?: string;
  primarySymptom?: string;
  boundingBoxes?: BoundingBox[];
}

interface Agency {
  name: string;
  location: string;
  contact: string;
}

enum DetectionStatus {
  IDLE = 'IDLE',
  ANALYZING = 'ANALYZING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}

// --- Constants ---
const LAGUNA_AGENCIES: Agency[] = [
  {
    name: "Laguna Lake Development Authority (LLDA)",
    location: "Quezon City / Laguna Regional Offices",
    contact: "(02) 8376-4039",
  },
  {
    name: "Bureau of Fisheries and Aquatic Resources (BFAR) Region 4A",
    location: "Los Baños, Laguna",
    contact: "(049) 536-8200",
  },
  {
    name: "DENR - PENRO Laguna",
    location: "Santa Cruz, Laguna",
    contact: "(049) 501-1424",
  },
  {
    name: "PDRRMO Laguna",
    location: "Santa Cruz, Laguna",
    contact: "911 / (049) 501-1254",
  }
];

const HAB_INFO_URL = "https://r7.bfar.da.gov.ph/red-tide/";
const ROBOFLOW_API_KEY = import.meta.env.VITE_ROBOFLOW_API_KEY || "6JUDsfKEWcpncQBxqbfY";
const MODEL_ENDPOINT = "https://detect.roboflow.com/liya-y6tnm/11";

// --- Helper Functions ---
function normalizeCoordinates(
  pred: { x: number; y: number; width: number; height: number },
  imgWidth: number,
  imgHeight: number
): BoundingBox {
  const xmin = ((pred.x - pred.width / 2) / imgWidth) * 1000;
  const xmax = ((pred.x + pred.width / 2) / imgWidth) * 1000;
  const ymin = ((pred.y - pred.height / 2) / imgHeight) * 1000;
  const ymax = ((pred.y + pred.height / 2) / imgHeight) * 1000;

  return {
    ymin: Math.max(0, Math.min(1000, ymin)),
    xmin: Math.max(0, Math.min(1000, xmin)),
    ymax: Math.max(0, Math.min(1000, ymax)),
    xmax: Math.max(0, Math.min(1000, xmax)),
  };
}

async function analyzeImage(base64Image: string): Promise<DetectionResult> {
  const cleanBase64 = base64Image.includes(',') ? base64Image.split(',')[1] : base64Image;

  try {
    const response = await fetch(`${MODEL_ENDPOINT}?api_key=${ROBOFLOW_API_KEY}&confidence=25`, {
      method: "POST",
      body: cleanBase64,
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

    if (!response.ok) throw new Error(`Roboflow API error: ${response.statusText}`);

    const data = await response.json();
    const predictions = data.predictions || [];
    const imgWidth = data.image?.width || 1;
    const imgHeight = data.image?.height || 1;

    const hasDetections = predictions.length > 0;
    const bestPred = hasDetections ? predictions.sort((a: any, b: any) => b.confidence - a.confidence)[0] : null;

    let status: DetectionResult['status'] = 'No Fish Detected';
    if (hasDetections) {
      const className = bestPred.class.toLowerCase();
      status = className.includes('healthy') ? 'Safe' : 'Contaminated';
    }

    return {
      status,
      confidence: bestPred ? bestPred.confidence : 0,
      location: "Binangonan, Rizal",
      primarySymptom: bestPred ? bestPred.class.replace(/-/g, ' ').toUpperCase() : "NONE",
      description: hasDetections 
        ? (status === 'Safe' 
            ? `The AI model identified the sample as ${bestPred.class} with ${Math.round(bestPred.confidence * 100)}% confidence. No HAB indicators were found.`
            : `Your custom model detected ${bestPred.class} with ${Math.round(bestPred.confidence * 100)}% certainty. This indicates a high risk of HAB contamination.`)
        : "No fish or HAB indicators were detected in the frame. Please ensure the sample is centered and well-lit.",
      guidance: hasDetections 
        ? (status === 'Safe'
            ? "Sample appears normal. However, always stay updated with the latest BFAR Red Tide Bulletins before consumption."
            : "DO NOT CONSUME. Report this sample to the nearest BFAR or LLDA station immediately.")
        : "Please try scanning again. Position the fish clearly in the center of the frame.",
      boundingBoxes: predictions.map((p: any) => normalizeCoordinates(p, imgWidth, imgHeight)),
      timestamp: new Date().toLocaleString('en-US', { 
        month: '2-digit', day: '2-digit', year: '2-digit', 
        hour: 'numeric', minute: '2-digit', hour12: true 
      }),
      image: base64Image
    };
  } catch (error) {
    console.error("Roboflow Inference failed:", error);
    throw new Error("Detection failed. Please check your Roboflow API configuration.");
  }
}

// --- Components ---

const CameraView: React.FC<{
  onCapture: (imageData: string) => void;
  isAnalyzing: boolean;
  lastCapturedImage: string | null;
  onToggleFlash: (isOn: boolean) => void;
  isFlashOn: boolean;
}> = ({ onCapture, isAnalyzing, lastCapturedImage, onToggleFlash, isFlashOn }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const constraints = {
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      };
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);
      if (videoRef.current) videoRef.current.srcObject = mediaStream;
    } catch (err) {
      console.error("Error accessing camera:", err);
    }
  }, []);

  useEffect(() => {
    startCamera();
    return () => stream?.getTracks().forEach(track => track.stop());
  }, [startCamera]);

  useEffect(() => {
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    const capabilities = track.getCapabilities() as any;
    if (capabilities.torch) {
      track.applyConstraints({ advanced: [{ torch: isFlashOn }] } as any).catch(e => console.warn("Flash failed:", e));
    }
  }, [isFlashOn, stream]);

  const capture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const context = canvasRef.current.getContext('2d');
    if (!context) return;
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;
    context.drawImage(videoRef.current, 0, 0);
    onCapture(canvasRef.current.toDataURL('image/jpeg'));
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="relative w-full aspect-[4/3] bg-black shadow-lg overflow-hidden">
        <video ref={videoRef} autoPlay playsInline className={`w-full h-full object-cover transition-opacity duration-300 ${isAnalyzing ? 'opacity-0' : 'opacity-100'}`} />
        {isAnalyzing && lastCapturedImage && <img src={lastCapturedImage} className="absolute inset-0 w-full h-full object-cover" alt="Analyzing" />}
        <canvas ref={canvasRef} className="hidden" />
        <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-white/50 m-4 z-10"></div>
        <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-white/50 m-4 z-10"></div>
        <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-white/50 m-4 z-10"></div>
        <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-white/50 m-4 z-10"></div>
        {isAnalyzing && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/20 text-white z-20">
            <div className="absolute inset-0 overflow-hidden pointer-events-none"><div className="animate-scan"></div></div>
            <div className="bg-black/60 backdrop-blur-md px-6 py-4 rounded-2xl flex flex-col items-center border border-white/10">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-cyan-400 mb-3"></div>
              <p className="font-bold tracking-[0.3em] text-sm text-cyan-400">SCANNING...</p>
            </div>
          </div>
        )}
      </div>
      <style>{`
        @keyframes scan { 0% { top: 0; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
        .animate-scan { position: absolute; width: 100%; height: 2px; background: linear-gradient(to right, transparent, #22d3ee, transparent); box-shadow: 0 0 15px #22d3ee; animation: scan 2s linear infinite; z-index: 30; }
      `}</style>
      <div className="flex items-center justify-around py-6 bg-white shadow-inner rounded-b-3xl">
        <button onClick={() => fileInputRef.current?.click()} className="w-14 h-14 bg-gray-100 rounded-xl overflow-hidden border-2 border-gray-200 shadow-sm flex items-center justify-center relative group">
          {lastCapturedImage ? <img src={lastCapturedImage} className="w-full h-full object-cover" alt="Last" /> : <span className="text-2xl">📁</span>}
          <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) {
              const reader = new FileReader();
              reader.onload = (ev) => onCapture(ev.target?.result as string);
              reader.readAsDataURL(file);
            }
          }} />
        </button>
        <button onClick={capture} disabled={isAnalyzing} className="relative flex items-center justify-center w-20 h-20 active:scale-90 disabled:opacity-50">
          <div className="absolute inset-0 border-[4px] border-gray-800 rounded-full"></div>
          <div className="w-16 h-16 bg-white border-2 border-gray-100 rounded-full shadow-inner"></div>
        </button>
        <div className="w-14 h-14 opacity-0"></div>
      </div>
    </div>
  );
};

const ResultDisplay: React.FC<{
  result: DetectionResult;
  onReset: () => void;
  onDownload: () => void;
}> = ({ result, onReset, onDownload }) => {
  const isContaminated = result.status === 'Contaminated';
  const isNoFish = result.status === 'No Fish Detected';

  const getZoomStyle = () => {
    if (isNoFish || !result.boundingBoxes || result.boundingBoxes.length === 0) return { transformOrigin: '50% 50%' };
    const box = result.boundingBoxes[0];
    return { transformOrigin: `${(box.xmin + box.xmax) / 20}% ${(box.ymin + box.ymax) / 20}%` };
  };

  if (isNoFish) {
    return (
      <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-lg mx-auto bg-white min-h-screen flex flex-col">
        <header className="px-6 py-4 flex items-center justify-between border-b border-cyan-50">
          <h1 className="text-3xl font-bold text-gray-900 tracking-widest">LIYA</h1>
        </header>
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
          <div className="w-24 h-24 bg-gray-50 rounded-full flex items-center justify-center mb-6 border-2 border-dashed border-gray-200"><span className="text-4xl">🔍</span></div>
          <h2 className="text-2xl font-black text-gray-900 mb-4 tracking-tight">NO FISH DETECTED</h2>
          <p className="text-gray-500 text-sm leading-relaxed mb-8 max-w-[280px]">{result.description}</p>
          <div className="w-full aspect-[4/3] rounded-2xl overflow-hidden border-2 border-gray-100 mb-8 shadow-sm grayscale opacity-60">
            <img src={result.image} className="w-full h-full object-cover" alt="Captured" />
          </div>
          <button onClick={onReset} className="w-full bg-black text-white py-4 rounded-2xl font-bold tracking-widest shadow-xl active:scale-95 transition-transform">SCAN AGAIN</button>
        </div>
        <div className="p-6 border-t border-gray-50 bg-gray-50/50">
          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">Tips for better detection:</p>
          <ul className="text-[10px] text-gray-500 space-y-1 text-left list-disc pl-4">
            <li>Ensure the fish is centered in the frame</li>
            <li>Use the flash in low-light conditions</li>
            <li>Avoid blurry or shaky captures</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-lg mx-auto bg-white min-h-screen pb-24">
      <header className="px-6 py-4 flex items-center justify-between border-b border-cyan-50">
        <h1 className="text-3xl font-bold text-gray-900 tracking-widest">LIYA</h1>
      </header>
      <div className="relative w-full aspect-[4/3] bg-black overflow-hidden shadow-inner">
        <img src={result.image} className="w-full h-full object-cover" alt="Result" />
        <div className="absolute top-0 left-0 p-3 text-white text-[10px] w-full flex justify-between font-medium">
          <span>{result.location}</span>
          <span>{result.timestamp}</span>
        </div>
        {result.boundingBoxes?.map((box, idx) => (
          <div key={idx} className="absolute border-2 border-cyan-400 pointer-events-none" style={{ top: `${box.ymin / 10}%`, left: `${box.xmin / 10}%`, width: `${(box.xmax - box.xmin) / 10}%`, height: `${(box.ymax - box.ymin) / 10}%` }}>
            <div className="absolute -top-6 left-0 bg-cyan-400 text-black text-[8px] font-bold px-1 whitespace-nowrap">{result.primarySymptom}</div>
          </div>
        ))}
      </div>
      <div className="p-6 space-y-6">
        <div className="flex flex-col items-center relative">
          <div className="absolute top-0 right-0 text-right">
            <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">Confidence</p>
            <p className="text-lg font-black text-gray-800">{(result.confidence * 100).toFixed(1)}%</p>
          </div>
          <div className={`px-12 py-2 rounded-full border-2 font-bold text-2xl tracking-widest ${isContaminated ? 'bg-red-50 text-red-600 border-red-200' : 'bg-emerald-50 text-emerald-600 border-emerald-200'}`}>
            {result.status.toUpperCase()}
          </div>
          <p className={`text-xs font-bold mt-2 uppercase tracking-tight ${isContaminated ? 'text-red-500' : 'text-emerald-500'}`}>
            {isContaminated ? 'Harmful Algal Bloom Detected' : 'No HAB Indicators Detected'}
          </p>
        </div>
        <div className="grid grid-cols-12 gap-5 items-start">
          <div className="col-span-5">
            <div className={`aspect-square border-[3px] overflow-hidden shadow-lg ${isContaminated ? 'border-red-600' : 'border-emerald-600'}`}>
              <img src={result.image} className="w-full h-full object-cover scale-[4]" style={getZoomStyle()} alt="Zoomed" />
            </div>
          </div>
          <div className="col-span-7 flex flex-col gap-2">
            <div className={`flex items-center gap-2 font-bold text-xl ${isContaminated ? 'text-red-600' : 'text-emerald-600'}`}>
              <span className="text-2xl">{isContaminated ? '❕' : '✅'}</span>
              {result.primarySymptom}
            </div>
            <p className="text-[11px] leading-relaxed text-gray-800 font-medium">{result.description}</p>
          </div>
        </div>
        <div className="space-y-4">
          <div className={`flex items-center gap-2 font-bold text-base ${isContaminated ? 'text-red-600' : 'text-emerald-600'}`}>
            <span className="text-xl">{isContaminated ? '🚫' : 'ℹ️'}</span>
            Safety Guidance: {result.guidance}
          </div>
          <div className="grid grid-cols-1 gap-4">
            <p className="text-xs font-bold text-gray-900 uppercase tracking-wide border-b border-gray-100 pb-1">Regional Reporting Agencies:</p>
            {LAGUNA_AGENCIES.map((agency, idx) => (
              <div key={idx} className="flex justify-between items-start text-[10px] gap-4">
                <div className="flex-1"><p className="font-bold text-gray-800">{agency.name}</p><p className="text-gray-500 italic">{agency.location}</p></div>
                <div className="text-right font-black text-gray-900 whitespace-nowrap">{agency.contact}</div>
              </div>
            ))}
          </div>
        </div>
        <button onClick={onDownload} className="w-full bg-[#4CAF50] text-white py-3 rounded-xl font-bold text-xs shadow-lg active:scale-95 transition-transform flex items-center justify-center gap-2">SAVE REPORT</button>
      </div>
      <button onClick={onReset} className="fixed bottom-24 right-6 w-14 h-14 bg-black text-white rounded-full shadow-2xl flex items-center justify-center text-[10px] font-black z-40 active:scale-90 transition-transform tracking-widest">SCAN</button>
      <div className="fixed bottom-0 left-0 w-full bg-yellow-50/95 backdrop-blur-sm py-4 flex justify-center items-center gap-2 border-t border-yellow-100 cursor-pointer z-30">
        <span className="text-xl">💡</span>
        <a href={HAB_INFO_URL} target="_blank" rel="noopener noreferrer" className="font-bold text-gray-900 text-base">Learn About HABs</a>
      </div>
    </div>
  );
};

// --- Main App ---

const App: React.FC = () => {
  const [status, setStatus] = useState<DetectionStatus>(DetectionStatus.IDLE);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastImg, setLastImg] = useState<string | null>(null);
  const [isFlashOn, setIsFlashOn] = useState(false);

  const handleCapture = async (imageData: string) => {
    setLastImg(imageData);
    setStatus(DetectionStatus.ANALYZING);
    setError(null);
    try {
      const analysisResult = await analyzeImage(imageData);
      setResult(analysisResult);
      setStatus(DetectionStatus.SUCCESS);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during analysis.');
      setStatus(DetectionStatus.ERROR);
    }
  };

  const resetDetection = () => {
    setResult(null);
    setStatus(DetectionStatus.IDLE);
    setError(null);
  };

  const downloadReport = () => {
    if (!result) return;
    const reportContent = `LIYA - HAB DETECTION REPORT\n==========================\nDate/Time: ${result.timestamp}\nLocation: ${result.location}\nOverall Status: ${result.status}\nConfidence: ${(result.confidence * 100).toFixed(1)}%\n\nINDICATORS DETECTED:\n- Primary Indicator: ${result.primarySymptom}\n- Description: ${result.description}\n\nGUIDANCE:\n- ${result.guidance}\n\nCONTACTS:\n- LLDA: (02) 8376-4039\n- BFAR: (049) 536-8200\n==========================`;
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `LIYA_Report_${new Date().getTime()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-md mx-auto min-h-screen flex flex-col shadow-2xl bg-white overflow-hidden relative">
      {!result && (
        <header className="px-6 py-6 flex items-center justify-between border-b border-cyan-50">
          <button onClick={() => setIsFlashOn(!isFlashOn)} className={`w-8 h-8 flex items-center justify-center transition-colors ${isFlashOn ? 'text-yellow-500' : 'text-gray-900'}`}>
            <svg className="w-8 h-8" viewBox="0 0 24 24" fill={isFlashOn ? "currentColor" : "none"} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
          </button>
          <h1 className="text-4xl font-bold text-gray-900 tracking-widest">LIYA</h1>
          <div className="w-8 h-8 flex flex-col justify-center gap-1.5"><div className="w-full h-0.5 bg-black"></div><div className="w-full h-0.5 bg-black"></div><div className="w-full h-0.5 bg-black"></div></div>
        </header>
      )}
      <main className="flex-1 overflow-y-auto">
        {result ? (
          <ResultDisplay result={result} onReset={resetDetection} onDownload={downloadReport} />
        ) : (
          <div className="p-4 pt-10">
            <CameraView onCapture={handleCapture} isAnalyzing={status === DetectionStatus.ANALYZING} lastCapturedImage={lastImg} isFlashOn={isFlashOn} onToggleFlash={setIsFlashOn} />
            {error && <div className="mt-4 p-4 bg-red-50 text-red-700 text-sm rounded-xl border border-red-100 flex items-center gap-3">⚠️ {error}</div>}
            <div className="mt-12 text-center text-gray-400">
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] mb-3 text-gray-600">Instructions</p>
              <p className="text-xs leading-relaxed max-w-[240px] mx-auto">Position the fish sample clearly in the center. Ensure light is sufficient for accurate AI analysis.</p>
            </div>
          </div>
        )}
      </main>
      {!result && (
        <a href={HAB_INFO_URL} target="_blank" rel="noopener noreferrer" className="py-4 flex justify-center items-center gap-2 border-t border-cyan-100 bg-cyan-50 hover:bg-cyan-100 transition-colors cursor-pointer">
          <span className="text-lg">💡</span><span className="font-semibold text-gray-800 text-sm">Learn About HABs</span>
        </a>
      )}
    </div>
  );
};

export default App;
