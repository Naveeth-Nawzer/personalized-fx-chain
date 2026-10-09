# Inverse FX Chain Predictor

A machine learning research project for automatically identifying the audio effects applied to a dry audio recording by analysing paired dry and processed (wet) audio.

This project is an individual research component of a larger intelligent audio-effects generation system. The long-term goal is to predict the structure of an audio-effects chain, including:

- Effect types
- Effect processing order
- Effect parameters

The current development stage focuses only on **effect-type prediction** using a **MERT-based audio representation**.

---

## Research Context

In music production, achieving a desired sound often requires selecting appropriate audio effects, determining their processing order, and tuning their parameters. This process can be time-consuming and requires technical expertise.

The proposed **Inverse FX Chain Predictor** treats this as an inverse audio-processing problem:

> Given a dry audio recording and its corresponding processed audio, infer the FX chain responsible for the transformation.

The complete research system is intended to eventually predict:

```text
Dry Audio + Processed Audio
            ↓
   Inverse FX Chain Predictor
            ↓
   ┌────────┼──────────┐
   ↓        ↓          ↓
Effects    Order    Parameters