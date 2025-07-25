## Rationale Block

**Pre-state**: The AgentManager implemented a simple ReAct loop (Think -> Act -> Observe) without strategic planning or self-reflection capabilities.

**Limitation**: The agent could only reason one step at a time, leading to inefficient paths and inability to handle complex multi-step goals that require foresight and adaptation.

**Decision Path**:
- Enhanced the cognitive architecture with explicit Planning and Reflection steps
- Modified AgentState to track plan execution with `current_plan_step_index` and `plan_steps_status`
- Created prompt templates for strategic planning and critical evaluation
- Restructured the main loop to: Plan -> (Think -> Act -> Observe -> Reflect)

**Post-state**: The agent now formulates a strategic plan before execution, maintains awareness of its current step, and critically evaluates progress after each action, enabling self-correction and adaptation.

## Commit Message

```text
feat(agents): implement planning and reflection for strategic agent cognition

* **Why** – Transform reactive agent into strategic problem-solver with foresight and self-correction
* **What** – Enhanced AgentState model, added _plan_step and _reflect_step methods, restructured main execution loop
* **Impact** – Agents now create strategic plans, track progress, and adapt based on reflection
* **Risk** – Increases LLM calls per iteration; robust error handling and fallbacks implemented

Refs: spec7_step-2.2.md
```

## External Behavior Changes

- **Enhanced Cognitive Architecture**: Agents now follow Plan -> (Think -> Act -> Observe -> Reflect) cycle instead of simple Think -> Act -> Observe
- **Strategic Planning**: Agents decompose complex goals into numbered action steps before execution
- **Progress Tracking**: Agents maintain awareness of current plan step and completion status
- **Self-Reflection**: After each action, agents critically evaluate progress and may update their plan
- **Improved Context**: The Think step now includes plan context, making tactical decisions more strategic
- **Better Observability**: Enhanced tracing and logging for plan execution and reflection

The enhanced system maintains backward compatibility while providing significantly more intelligent and strategic agent behavior for complex multi-step tasks.
