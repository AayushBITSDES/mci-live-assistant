"""Onboarding: persona setup, emergency contacts, demographics.

Maps to Design Decision #9 (collects emergency contacts, weight, height,
age — explained to user during onboarding).
"""
from __future__ import annotations

import streamlit as st

from app.config import settings
from dashboard.persistence import EmergencyContact, OnboardingProfile, load_onboarding, save_onboarding

st.set_page_config(page_title="Onboarding", page_icon="👤", layout="wide")
st.title("👤 Onboarding")
st.caption(
    "We collect this so the system can help in an emergency. "
    "It is stored locally on this machine only — never uploaded."
)

settings.ensure_dirs()

profile = load_onboarding(settings.onboarding_file)

# --- Identity -------------------------------------------------------------

st.subheader("Persona")
col1, col2 = st.columns(2)
new_user = col1.text_input("User name", value=profile.user_name)
new_persona = col2.selectbox(
    "Persona",
    options=["shanta"],          # Sanjay deprioritized per May 2026 decision
    index=0,
)

st.subheader("Demographics")
col3, col4, col5, col6 = st.columns(4)
new_age = col3.number_input("Age", min_value=0, max_value=130, value=profile.age or 67)
new_mci = col4.selectbox(
    "MCI status",
    options=["none", "mild", "moderate"],
    index=["none", "mild", "moderate"].index(profile.mci_status if profile.mci_status in ["none", "mild", "moderate"] else "mild"),
)
new_weight = col5.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=profile.weight_kg or 0.0, step=0.5)
new_height = col6.number_input("Height (cm)", min_value=0.0, max_value=300.0, value=profile.height_cm or 0.0, step=1.0)

# --- Emergency contacts ---------------------------------------------------

st.markdown("---")
st.subheader("Emergency contacts")
st.caption("In a serious situation the system may suggest calling these people.")

if "emergency_contacts" not in st.session_state:
    st.session_state["emergency_contacts"] = list(profile.emergency_contacts)

contacts = st.session_state["emergency_contacts"]

for i, contact in enumerate(contacts):
    cols = st.columns([3, 3, 4, 1])
    contact.name = cols[0].text_input(f"Name #{i+1}", value=contact.name, key=f"name-{i}")
    contact.relationship = cols[1].text_input(f"Relationship #{i+1}", value=contact.relationship, key=f"rel-{i}")
    contact.phone = cols[2].text_input(f"Phone #{i+1}", value=contact.phone, key=f"phone-{i}")
    if cols[3].button("✕", key=f"del-{i}"):
        contacts.pop(i)
        st.rerun()

if st.button("+ Add contact"):
    contacts.append(EmergencyContact(name="", relationship="", phone=""))
    st.rerun()

# --- Notes ----------------------------------------------------------------

st.markdown("---")
st.subheader("Notes")
new_notes = st.text_area(
    "Anything else the system should know? (medications, routines, preferences)",
    value=profile.notes,
    height=100,
)

# --- Save -----------------------------------------------------------------

st.markdown("---")
if st.button("💾 Save profile"):
    cleaned_contacts = [
        c for c in contacts
        if c.name.strip() or c.phone.strip()
    ]
    new_profile = OnboardingProfile(
        persona=new_persona,
        user_name=new_user,
        age=int(new_age) if new_age else None,
        mci_status=new_mci,
        weight_kg=float(new_weight) if new_weight > 0 else None,
        height_cm=float(new_height) if new_height > 0 else None,
        emergency_contacts=cleaned_contacts,
        notes=new_notes,
    )
    save_onboarding(new_profile, settings.onboarding_file)
    st.success("Profile saved.")
