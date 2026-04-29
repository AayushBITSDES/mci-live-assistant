import type { ScenarioDefinition } from "./types";

export const scenarios: Record<string, ScenarioDefinition> = {
  "b2-shanta-blood-test": {
    id: "b2-shanta-blood-test",
    code: "B2",
    persona: "Shanta",
    title: "Blood test departure window",
    summary:
      "Shanta has a 10am blood test and moves through her morning routine without recalling the appointment.",
    targetSurface: "quest",
    context: [
      "Shanta is 67 and uses the assistant as an ambient smart-glasses support layer.",
      "Her blood test is at 10:00am and the clinic is a short trip away.",
      "The useful moment is not the appointment time; it is the moment she must leave."
    ],
    plannedNudge: "You have a blood test at 10am, and it is time to leave now.",
    steps: [
      {
        id: "wake",
        title: "Morning routine begins",
        ambientDetail: "Shanta is in the kitchen and starts her normal morning routine.",
        event: "routine_started",
        riskScore: 0.15
      },
      {
        id: "routine",
        title: "Routine absorbs attention",
        ambientDetail: "She prepares tea and tidies the counter without mentioning the appointment.",
        event: "routine_started",
        riskScore: 0.35
      },
      {
        id: "leave-soon",
        title: "Leaving window approaches",
        ambientDetail: "The calendar says she should be getting ready to leave for the clinic.",
        event: "risk_window",
        riskScore: 0.72
      },
      {
        id: "leave-now",
        title: "Optimal nudge moment",
        ambientDetail: "She is still at home and the departure window is now active.",
        event: "risk_window",
        riskScore: 0.92
      },
      {
        id: "missed-window",
        title: "Risk escalates",
        ambientDetail: "The window to leave is passing and the assistant should repeat if unresolved.",
        event: "risk_window",
        riskScore: 0.98
      }
    ]
  },
  "b1-sanjay-client-deck": {
    id: "b1-sanjay-client-deck",
    code: "B1",
    persona: "Sanjay",
    title: "Client deck before 3pm meeting",
    summary:
      "Sanjay needs to send a client deck before a 3pm meeting but gets buried in Slack threads.",
    targetSurface: "mobile",
    context: [
      "Sanjay is a 34-year-old product manager.",
      "The client deck must go out before a 3pm meeting.",
      "Slack distraction is the candidate trigger, but full detection is deferred."
    ],
    plannedNudge: "Send the client deck now so the client has it before the 3pm meeting.",
    steps: [
      {
        id: "intent",
        title: "Intent captured",
        ambientDetail: "The deck is on Sanjay's task list for the 3pm client meeting.",
        event: "routine_started",
        riskScore: 0.25
      },
      {
        id: "slack-load",
        title: "Slack distraction shell",
        ambientDetail: "Back-to-back Slack threads consume the morning.",
        event: "interruption",
        riskScore: 0.56
      },
      {
        id: "meeting-near",
        title: "Meeting window shell",
        ambientDetail: "The meeting is nearing and the deck is still marked unsent.",
        event: "risk_window",
        riskScore: 0.78
      }
    ]
  }
};

export const defaultScenarioId = "b2-shanta-blood-test";
