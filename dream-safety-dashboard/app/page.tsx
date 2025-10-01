"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Shield,
  Camera,
  MapPin,
  Clock,
  Users,
  AlertTriangle,
  Phone,
  Lock,
  Activity,
  Cpu,
  HardDrive,
  Wifi,
  Eye,
  Navigation,
  VolumeX,
  UserCheck,
  UserX,
  CircleX,
  CheckCircle,
} from "lucide-react"

interface Detection {
  persons: Array<{
    name: string
    confidence: number
    bbox: number[]
    is_threat: boolean
    location: string
  }>
  weapons: Array<{
    type: string
    weapon_type: string
    confidence: number
    bbox: number[]
    location: string
    timestamp: string
  }>
  threat_count: number
}

interface Threat {
  type: string
  person?: string
  weapon_type?: string
  confidence: number
  location: string
  timestamp: string
}

interface DetectionData {
  timestamp: string
  location: string
  frame: string
  detections: Detection
  threats: Threat[]
  status: "alert" | "clear"
}

export default function DreamSafetyDashboard() {
  const [currentTime, setCurrentTime] = useState(new Date())
  const [incidentTimer, setIncidentTimer] = useState(0)
  const [policeETA, setPoliceETA] = useState(180) // seconds
  const [focusedFeed, setFocusedFeed] = useState(0)
  const [mapFocused, setMapFocused] = useState(false)
  const [showLockdownConfirm, setShowLockdownConfirm] = useState(false)
  const [showNotifications, setShowNotifications] = useState(false)
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false)
  
  // WebSocket related states
  const [isConnected, setIsConnected] = useState(false)
  const [detectionData, setDetectionData] = useState<DetectionData | null>(null)
  const [cameraFeeds, setCameraFeeds] = useState<Array<{ location: string; frame: string; status: string; threats: number }>>([])
  const [threatTimeline, setThreatTimeline] = useState<Array<{
    time: string
    event: string
    location: string
    severity: "high" | "medium" | "low"
  }>>([])
  const [activeThreat, setActiveThreat] = useState<Threat | null>(null)
  const [notifications, setNotifications] = useState<Array<{
    id: number
    severity: "high" | "medium" | "low"
    message: string
    time: string
    type: string
  }>>([])
  
  // Pathfinding related states
  const [currentFloor, setCurrentFloor] = useState("Ground")
  const [floorPlanImage, setFloorPlanImage] = useState<string | null>(null)
  const [pathInstructions, setPathInstructions] = useState<string[]>([])
  const [pathDistance, setPathDistance] = useState(0)
  const [evacuationRoutes, setEvacuationRoutes] = useState<Array<{
    destination: string
    distance: number
    instructions: string[]
  }>>([])
  
  const wsRef = useRef<WebSocket | null>(null)
  const notificationIdRef = useRef(1)

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket("ws://localhost:8765")
        wsRef.current = ws

        ws.onopen = () => {
          console.log("Connected to detection server")
          setIsConnected(true)
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            
            // Handle different message types
            if (data.type === "detection_data") {
              const detectionData: DetectionData = data
              setDetectionData(detectionData)
              
              // Update camera feeds
              setCameraFeeds(prev => {
                const existing = prev.findIndex(f => f.location === detectionData.location)
                const feed = {
                  location: detectionData.location,
                  frame: detectionData.frame,
                  status: detectionData.status,
                  threats: detectionData.detections.threat_count
                }
                
                if (existing >= 0) {
                  const updated = [...prev]
                  updated[existing] = feed
                  return updated
                } else {
                  return [...prev, feed].slice(-4) // Keep only last 4 feeds
                }
              })
              
              // Process threats
              if (detectionData.threats.length > 0) {
                // Set incident timer if first threat
                if (incidentTimer === 0) {
                  setIncidentTimer(1)
                }
                
                // Update active threat
                const highestPriorityThreat = detectionData.threats.find(t => t.type === "armed_person") || detectionData.threats[0]
                setActiveThreat(highestPriorityThreat)
                
                // Add to timeline
                detectionData.threats.forEach(threat => {
                  const timeStr = new Date(threat.timestamp).toLocaleTimeString()
                  const eventMessage = threat.type === "armed_person" 
                    ? `Armed person: ${threat.person}`
                    : threat.type === "weapon_detected"
                    ? `Weapon detected: ${threat.weapon_type}`
                    : "Unknown threat"
                  
                  setThreatTimeline(prev => {
                    const exists = prev.some(t => 
                      t.time === timeStr && t.event === eventMessage
                    )
                    if (!exists) {
                      const newEvent: { time: string; event: string; location: string; severity: "high" | "medium" | "low" } = {
                        time: timeStr,
                        event: eventMessage,
                        location: threat.location,
                        severity: threat.confidence > 0.8 ? "high" : threat.confidence > 0.5 ? "medium" : "low"
                      };
                      return [newEvent, ...prev].slice(0, 10); // Keep last 10 events

                    }
                    return prev
                  })
                  
                  // Add notification
                  if (threat.confidence > 0.7) {
                    const notif = {
                      id: notificationIdRef.current++,
                      severity: threat.confidence > 0.8 ? "high" as const : "medium" as const,
                      message: eventMessage,
                      time: timeStr,
                      type: threat.type
                    }
                    setNotifications(prev => [notif, ...prev].slice(0, 20))
                  }
                })
              }
            }
            // Handle pathfinding responses
            else if (data.type === "pathfinding_response") {
              if (data.request_type === "intercept") {
                const result = data.data
                if (result.status === "success") {
                  setFloorPlanImage(result.floor_plan)
                  setPathInstructions(result.instructions)
                  setPathDistance(result.distance)
                  
                  // Add notification
                  const notif = {
                    id: notificationIdRef.current++,
                    severity: "medium" as const,
                    message: `Intercept path calculated for ${result.officer}`,
                    time: new Date().toLocaleTimeString(),
                    type: "pathfinding"
                  }
                  setNotifications(prev => [notif, ...prev].slice(0, 20))
                }
              } else if (data.request_type === "evacuation") {
                setEvacuationRoutes(data.data.routes)
                
                // Add notification
                const notif = {
                  id: notificationIdRef.current++,
                  severity: "low" as const,
                  message: `Evacuation routes calculated from ${data.data.location}`,
                  time: new Date().toLocaleTimeString(),
                  type: "evacuation"
                }
                setNotifications(prev => [notif, ...prev].slice(0, 20))
              }
            }
            // Handle auto-intercept paths
            else if (data.type === "auto_intercept") {
              const result = data.data
              if (result.status === "success") {
                setFloorPlanImage(result.floor_plan)
                setPathInstructions(result.instructions)
                setPathDistance(result.distance)
                
                // Add notification with high priority
                const notif = {
                  id: notificationIdRef.current++,
                  severity: "high" as const,
                  message: `Auto-intercept: ${result.officer} → ${result.threat_location}`,
                  time: new Date().toLocaleTimeString(),
                  type: "auto_intercept"
                }
                setNotifications(prev => [notif, ...prev].slice(0, 20))
              }
            }
            
          } catch (error) {
            console.error("Error parsing WebSocket data:", error)
          }
        }

        ws.onerror = (error) => {
          console.error("WebSocket error:", error)
          setIsConnected(false)
        }

        ws.onclose = () => {
          console.log("Disconnected from detection server")
          setIsConnected(false)
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000)
        }

      } catch (error) {
        console.error("Failed to connect to WebSocket:", error)
        setIsConnected(false)
        // Retry connection after 3 seconds
        setTimeout(connectWebSocket, 3000)
      }
    }

    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Update timers
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
      if (incidentTimer > 0) {
        setIncidentTimer((prev) => prev + 1)
      }
      if (policeETA > 0 && incidentTimer > 0) {
        setPoliceETA((prev) => Math.max(0, prev - 1))
      }
    }, 1000)
    return () => clearInterval(timer)
  }, [incidentTimer, policeETA])

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

    switch (e.key) {
      case "1":
      case "2":
      case "3":
      case "4":
        e.preventDefault()
        setFocusedFeed(Number.parseInt(e.key) - 1)
        setMapFocused(false)
        break
      case "m":
      case "M":
        e.preventDefault()
        setMapFocused(!mapFocused)
        break
      case "l":
      case "L":
        e.preventDefault()
        setShowLockdownConfirm(true)
        break
      case "n":
      case "N":
        e.preventDefault()
        setShowNotifications(!showNotifications)
        break
      case "Escape":
        e.preventDefault()
        setShowLockdownConfirm(false)
        setShowNotifications(false)
        setShowKeyboardShortcuts(false)
        break
      case "?":
        e.preventDefault()
        setShowKeyboardShortcuts(!showKeyboardShortcuts)
        break
    }
  }

  useEffect(() => {
    window.addEventListener("keydown", handleKeyPress)
    return () => window.removeEventListener("keydown", handleKeyPress)
  }, [mapFocused, showNotifications, showKeyboardShortcuts])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  const formatETA = (seconds: number) => {
    if (seconds <= 0) return "On Site"
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")} ETA`
  }

  // Pathfinding functions
  const requestInterceptPath = (officerId: string) => {
    if (!wsRef.current || !activeThreat) return
    
    const request = {
      type: "pathfinding_request",
      request_type: "intercept",
      officer_id: officerId,
      threat_location: activeThreat.location
    }
    
    wsRef.current.send(JSON.stringify(request))
  }
  
  const requestEvacuationRoutes = () => {
    if (!wsRef.current || !activeThreat) return
    
    const request = {
      type: "pathfinding_request",
      request_type: "evacuation",
      location: activeThreat.location
    }
    
    wsRef.current.send(JSON.stringify(request))
  }
  
  const updateOfficerPosition = (officerId: string, position: string) => {
    if (!wsRef.current) return
    
    const request = {
      type: "pathfinding_request",
      request_type: "update_officer",
      officer_id: officerId,
      position: position
    }
    
    wsRef.current.send(JSON.stringify(request))
  }

  // Calculate student accounting based on detections
  const studentAccounting = {
    safe: detectionData ? 
      Math.max(1198 - (detectionData.detections.persons.filter(p => p.is_threat).length * 50), 1000) : 1198,
    missing: detectionData && detectionData.threats.length > 0 ? 
      Math.min(56 + (detectionData.threats.length * 10), 100) : 56,
    inTransit: 12
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {showKeyboardShortcuts && (
        <div className="fixed top-20 right-4 z-40 bg-white/95 backdrop-blur-sm border rounded-lg p-3 text-xs space-y-1 shadow-lg w-48">
          <div className="flex items-center justify-between mb-2">
            <div className="font-semibold">Shortcuts</div>
            <Button
              variant="ghost"
              size="sm"
              className="h-4 w-4 p-0 text-muted-foreground hover:text-foreground"
              onClick={() => setShowKeyboardShortcuts(false)}
            >
              ×
            </Button>
          </div>
          <div>
            <kbd className="bg-muted px-1 rounded">1-4</kbd> CCTV feeds
          </div>
          <div>
            <kbd className="bg-muted px-1 rounded">M</kbd> Map focus
          </div>
          <div>
            <kbd className="bg-muted px-1 rounded">L</kbd> Lockdown
          </div>
          <div>
            <kbd className="bg-muted px-1 rounded">N</kbd> Notifications
          </div>
          <div>
            <kbd className="bg-muted px-1 rounded">?</kbd> Help
          </div>
        </div>
      )}

      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Shield className="h-7 w-7 text-emerald-600" />
              <span className="text-xl font-bold text-slate-800">Dream Safety</span>
            </div>
            {incidentTimer > 0 && (
              <Badge variant="destructive" className="animate-pulse font-medium">
                LIVE INCIDENT
              </Badge>
            )}
            <div className="flex items-center gap-2">
              {isConnected ? (
                <CheckCircle className="h-4 w-4 text-emerald-500" />
              ) : (
                <CircleX className="h-4 w-4 text-red-500" />
              )}
              <span className="text-xs text-slate-600">
                {isConnected ? "Connected" : "Connecting..."}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 text-sm text-slate-600">
              <Clock className="h-4 w-4" />
              <span className="font-mono">{currentTime.toLocaleTimeString()}</span>
            </div>

            {incidentTimer > 0 && (
              <>
                <div className="flex items-center gap-2 text-sm">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <span className="font-mono text-red-600 font-medium">Incident: {formatTime(incidentTimer)}</span>
                </div>

                <div className="flex items-center gap-2 text-sm">
                  <Navigation className="h-4 w-4 text-emerald-600" />
                  <span className="font-mono text-emerald-600 font-medium">Police {formatETA(policeETA)}</span>
                </div>
              </>
            )}

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative"
            >
              <AlertTriangle className="h-4 w-4" />
              {notifications.length > 0 && (
                <Badge className="absolute -top-1 -right-1 h-5 w-5 p-0 text-xs bg-red-500">
                  {notifications.length}
                </Badge>
              )}
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowKeyboardShortcuts(!showKeyboardShortcuts)}
              className="text-slate-600"
            >
              ?
            </Button>
          </div>
        </div>
      </header>

      <div className="p-6">
        <div className="grid grid-cols-12 gap-6 h-[calc(100vh-140px)]">
          {/* Left Sidebar - Live CCTV Feeds */}
          <div className="col-span-3 space-y-4">
            <div>
              <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <Camera className="h-5 w-5" />
                Live CCTV Feeds
                <Badge variant="outline" className="text-xs">
                  Feed {focusedFeed + 1}
                </Badge>
              </h2>

              <div className="space-y-3">
                {cameraFeeds.length > 0 ? (
                  cameraFeeds.slice(0, 3).map((feed, index) => (
                    <Card
                      key={index}
                      className={`relative overflow-hidden cursor-pointer hover:shadow-md transition-all ${
                        focusedFeed === index ? "ring-2 ring-emerald-500 shadow-md" : ""
                      } ${feed.status === "alert" ? "ring-2 ring-red-500" : ""}`}
                      onClick={() => setFocusedFeed(index)}
                    >
                      <div className="aspect-[4/3] bg-slate-900 relative">
                        {feed.frame && (
                          <img
                            src={`data:image/jpeg;base64,${feed.frame}`}
                            alt={`CCTV ${feed.location}`}
                            className="w-full h-full object-cover"
                          />
                        )}
                        <div className="absolute top-2 left-2 bg-black/80 text-white text-xs px-2 py-1 rounded font-mono">
                          {feed.location}
                        </div>
                        <div className="absolute top-2 right-2">
                          <Badge className={`${
                            feed.status === "alert" ? "bg-red-500" : "bg-emerald-500"
                          } text-white text-xs`}>
                            {feed.status === "clear" ? "✓ Clear" : `! ${feed.threats} threats`}
                          </Badge>
                        </div>
                        <div className="absolute bottom-2 left-2 flex items-center gap-1">
                          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                          <span className="text-white text-xs bg-black/80 px-1 rounded">LIVE</span>
                        </div>
                        <div className="absolute bottom-2 right-2">
                          <kbd className="bg-black/80 text-white px-1 py-0.5 rounded text-xs">{index + 1}</kbd>
                        </div>
                        {feed.status === "alert" && (
                          <div className="absolute inset-0 border-2 border-red-500 animate-pulse"></div>
                        )}
                      </div>
                    </Card>
                  ))
                ) : (
                  <div className="text-center text-slate-500 py-8">
                    <Camera className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Waiting for camera feeds...</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Center - Map and Active Threat Profile */}
          <div className="col-span-6 space-y-4">
            {activeThreat && (
              <Card className="border-red-200 bg-red-50/50">
                <CardHeader className="pb-3">
                  <CardTitle className="text-red-700 flex items-center gap-2 text-xl">
                    <AlertTriangle className="h-6 w-6" />
                    Active Threat Profile
                    <Badge variant="destructive" className="ml-2">
                      PRIORITY 1
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden">
                      {detectionData && detectionData.frame && (
                        <img
                          src={`data:image/jpeg;base64,${detectionData.frame}`}
                          alt="Threat"
                          className="w-full h-full object-cover"
                        />
                      )}
                    </div>

                    <div className="col-span-2 space-y-3">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="font-semibold text-red-700">Location</div>
                          <div>{activeThreat.location}</div>
                        </div>
                        <div>
                          <div className="font-semibold text-red-700">Threat Type</div>
                          <div>{activeThreat.type.replace('_', ' ')}</div>
                        </div>
                        {activeThreat.weapon_type && (
                          <div>
                            <div className="font-semibold text-red-700">Weapon</div>
                            <div>{activeThreat.weapon_type} ({(activeThreat.confidence * 100).toFixed(0)}%)</div>
                          </div>
                        )}
                        {activeThreat.person && (
                          <div>
                            <div className="font-semibold text-red-700">Individual</div>
                            <div>{activeThreat.person}</div>
                          </div>
                        )}
                      </div>

                      <div className="flex gap-2">
                        <Button size="sm" className="flex-1 bg-red-600 hover:bg-red-700">
                          Broadcast Alert
                        </Button>
                        <Button size="sm" variant="outline" className="flex-1 bg-transparent">
                          Share Snapshot
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            <Card className={`flex-1 transition-all ${mapFocused ? "ring-2 ring-emerald-500 shadow-lg" : ""}`}>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="h-5 w-5" />
                  School Floor Plan - {currentFloor}
                  {mapFocused && (
                    <Badge variant="outline" className="ml-2 text-xs">
                      Focused
                    </Badge>
                  )}
                  <div className="ml-auto flex gap-1">
                    <Button 
                      size="sm" 
                      variant={currentFloor === "Ground" ? "outline" : "ghost"}
                      className="h-7 text-xs bg-transparent"
                      onClick={() => setCurrentFloor("Ground")}
                    >
                      Ground
                    </Button>
                    <Button 
                      size="sm" 
                      variant={currentFloor === "Floor 1" ? "outline" : "ghost"}
                      className="h-7 text-xs"
                      onClick={() => setCurrentFloor("Floor 1")}
                    >
                      Floor 1
                    </Button>
                    <Button 
                      size="sm" 
                      variant={currentFloor === "Floor 2" ? "outline" : "ghost"}
                      className="h-7 text-xs"
                      onClick={() => setCurrentFloor("Floor 2")}
                    >
                      Floor 2
                    </Button>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="h-full">
                <div className="bg-slate-100 rounded-lg h-full relative overflow-hidden min-h-[400px]">
                  {floorPlanImage ? (
                    <img
                      src={`data:image/jpeg;base64,${floorPlanImage}`}
                      alt="School Floor Plan with Path"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <img
                      src="/placeholder.svg?height=500&width=800"
                      alt="School Floor Plan"
                      className="w-full h-full object-cover"
                    />
                  )}
                  
                  {/* Path Instructions Overlay */}
                  {pathInstructions.length > 0 && (
                    <div className="absolute top-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 max-w-xs max-h-48 overflow-y-auto">
                      <div className="text-xs font-semibold mb-2 flex items-center gap-1">
                        <Navigation className="h-3 w-3" />
                        Navigation Instructions
                      </div>
                      <div className="space-y-1">
                        {pathInstructions.map((instruction, idx) => (
                          <div key={idx} className="text-xs flex items-start gap-1">
                            <span className="text-slate-400">{idx + 1}.</span>
                            <span>{instruction}</span>
                          </div>
                        ))}
                      </div>
                      {pathDistance > 0 && (
                        <div className="text-xs font-semibold mt-2 text-emerald-600">
                          Distance: {pathDistance.toFixed(0)}m
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Officer Controls */}
                  <div className="absolute bottom-4 right-4 bg-white/95 backdrop-blur-sm rounded-lg p-3">
                    <div className="text-xs font-semibold mb-2">Officer Controls</div>
                    <div className="space-y-2">
                      <Button 
                        size="sm" 
                        className="w-full text-xs bg-blue-600 hover:bg-blue-700"
                        onClick={() => requestInterceptPath("Officer1")}
                        disabled={!activeThreat}
                      >
                        Officer 1 → Intercept
                      </Button>
                      <Button 
                        size="sm" 
                        className="w-full text-xs bg-blue-600 hover:bg-blue-700"
                        onClick={() => requestInterceptPath("Officer2")}
                        disabled={!activeThreat}
                      >
                        Officer 2 → Intercept
                      </Button>
                      <Button 
                        size="sm" 
                        variant="outline"
                        className="w-full text-xs bg-transparent"
                        onClick={requestEvacuationRoutes}
                      >
                        Show Evacuation Routes
                      </Button>
                    </div>
                  </div>
                  
                  {/* Legend */}
                  <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm rounded-lg p-3 text-xs space-y-1">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <span>Active Threat</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <span>Officers</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-red-500"></div>
                      <span>Intercept Path</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-purple-500"></div>
                      <span>Stairs</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Sidebar - Timeline, Student Accounting and System Status */}
          <div className="col-span-3 space-y-4">
            {/* Threat Timeline */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Threat Timeline</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 max-h-64 overflow-y-auto">
                {threatTimeline.length > 0 ? (
                  threatTimeline.map((event, index) => (
                    <div
                      key={index}
                      className={`flex items-start gap-3 text-sm p-2 rounded cursor-pointer hover:bg-slate-50 ${
                        event.severity === "high" ? "bg-red-50 border-l-2 border-red-500" : ""
                      }`}
                    >
                      <AlertTriangle className={`h-4 w-4 mt-0.5 ${
                        event.severity === "high" ? "text-red-500" : "text-yellow-500"
                      }`} />
                      <div className="flex-1">
                        <div className="font-mono text-xs text-slate-500">{event.time}</div>
                        <div className="font-medium">{event.event}</div>
                        <div className="text-xs text-slate-600">{event.location}</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-slate-500 py-4">
                    <p className="text-sm">No threats detected</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Student Accounting */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5" />
                  Student Accounting
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-emerald-50 rounded-2xl p-4 border border-emerald-200">
                  <div className="flex items-center gap-3">
                    <UserCheck className="h-6 w-6 text-emerald-600" />
                    <div>
                      <div className="text-3xl font-bold text-emerald-700">{studentAccounting.safe}</div>
                      <div className="text-emerald-600 font-medium text-sm">Students Safe</div>
                    </div>
                  </div>
                </div>

                <div className="bg-orange-50 rounded-2xl p-4 border border-orange-200">
                  <div className="flex items-center gap-3">
                    <UserX className="h-6 w-6 text-orange-600" />
                    <div>
                      <div className="text-3xl font-bold text-orange-700">{studentAccounting.missing}</div>
                      <div className="text-orange-600 font-medium text-sm">Students Missing</div>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 rounded-2xl p-4 border border-blue-200">
                  <div className="flex items-center gap-3">
                    <Users className="h-6 w-6 text-blue-600" />
                    <div>
                      <div className="text-3xl font-bold text-blue-700">{studentAccounting.inTransit}</div>
                      <div className="text-blue-600 font-medium text-sm">In Transit</div>
                    </div>
                  </div>
                </div>

                <div className="pt-2">
                  <Progress value={95.5} className="h-3" />
                  <p className="text-xs text-slate-600 mt-2 flex items-center justify-between">
                    <span>95.5% accounted for</span>
                    <span className="text-emerald-600">Real-time tracking</span>
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* System Health */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Activity className="h-4 w-4" />
                  System Health
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Camera className="h-3 w-3 text-emerald-500" />
                      <span>Detection Server</span>
                    </div>
                    <Badge className={`${isConnected ? "bg-emerald-500" : "bg-red-500"} text-white text-xs`}>
                      {isConnected ? "Online" : "Offline"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Eye className="h-3 w-3 text-emerald-500" />
                      <span>AI Detection</span>
                    </div>
                    <Badge className="bg-emerald-500 text-white text-xs">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Cpu className="h-3 w-3 text-yellow-500" />
                      <span>Edge Processing</span>
                    </div>
                    <Badge className="bg-yellow-500 text-white text-xs">Processing</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <HardDrive className="h-3 w-3 text-emerald-500" />
                      <span>Storage</span>
                    </div>
                    <Badge className="bg-emerald-500 text-white text-xs">Recording</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Wifi className="h-3 w-3" />
                      <span>WebSocket</span>
                    </div>
                    <Badge className={`${isConnected ? "bg-emerald-500" : "bg-slate-500"} text-white text-xs`}>
                      {isConnected ? "Connected" : "Reconnecting..."}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      <div className="fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-sm border-t shadow-lg">
        <div className="flex justify-center items-center gap-4 p-4">
          <Button
            variant="destructive"
            size="lg"
            className="gap-2 font-semibold"
            onClick={() => setShowLockdownConfirm(true)}
            disabled={!isConnected}
          >
            <Lock className="h-4 w-4" />
            School Lockdown
          </Button>
          <Button variant="default" size="lg" className="gap-2 bg-orange-500 hover:bg-orange-600 font-semibold">
            <Phone className="h-4 w-4" />
            Call 911
          </Button>
          <Button variant="outline" size="lg" className="gap-2 font-semibold bg-transparent">
            <VolumeX className="h-4 w-4" />
            Mute Alarms
          </Button>
          <Button variant="secondary" size="lg" className="gap-2 font-semibold">
            <Shield className="h-4 w-4" />
            Emergency Drill
          </Button>
        </div>
      </div>

      {showLockdownConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-96 border-red-200">
            <CardHeader>
              <CardTitle className="text-red-700 flex items-center gap-2">
                <Lock className="h-5 w-5" />
                Confirm School Lockdown
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm">This will immediately secure all entrances and notify all staff. Are you sure?</p>
              <div className="flex gap-2">
                <Button
                  variant="destructive"
                  className="flex-1"
                  onClick={() => {
                    setShowLockdownConfirm(false)
                    // Handle lockdown action
                  }}
                >
                  Confirm Lockdown
                </Button>
                <Button
                  variant="outline"
                  className="flex-1 bg-transparent"
                  onClick={() => setShowLockdownConfirm(false)}
                >
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {showNotifications && (
        <div className="fixed right-4 top-20 w-96 max-h-96 z-40">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center justify-between">
                Notifications & Alerts
                <Button variant="ghost" size="sm" onClick={() => setShowNotifications(false)}>
                  ×
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 max-h-80 overflow-y-auto">
              {notifications.length > 0 ? (
                notifications.map((notification) => (
                  <div
                    key={notification.id}
                    className={`p-3 rounded-lg border-l-4 ${
                      notification.severity === "high" ? "border-red-500 bg-red-50" : "border-yellow-500 bg-yellow-50"
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="text-sm font-medium">{notification.message}</div>
                        <div className="text-xs text-muted-foreground font-mono">{notification.time}</div>
                      </div>
                      <div className="flex gap-1">
                        <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
                          <Eye className="h-3 w-3" />
                        </Button>
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          className="h-6 w-6 p-0"
                          onClick={() => setNotifications(prev => prev.filter(n => n.id !== notification.id))}
                        >
                          ×
                        </Button>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-slate-500 py-4">
                  <p className="text-sm">No notifications</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}