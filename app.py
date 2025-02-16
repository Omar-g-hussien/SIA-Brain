import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
import sqlalchemy
from sqlalchemy import inspect
import json
from datetime import datetime
from typing import Dict, Optional, Any

class QuerySystem:
    def __init__(self):
        self.engine = self._init_engine()
        self.db = SQLDatabase(self.engine)
        self.agent = self._create_agent()
        self._initialize_session_state()
        
    @staticmethod
    @st.cache_resource
    def _init_engine():
        return sqlalchemy.create_engine("sqlite:///data.db")
    
    @staticmethod
    @st.cache_resource
    def _create_agent():
        engine = QuerySystem._init_engine()
        db = SQLDatabase(engine)
        llm = OpenAI(temperature=0)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        return create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            max_iterations=None
        )
    
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        default_states = {
            "current_step": "start",
            "messages": [],
            "offers": self._load_data(),
            "active_category": None,
            "current_offer": {},
            "offer_step": 0,
            "rules": {
                "merchant": {"conditions": [], "count": 0},
                "transaction": {"conditions": [], "count": 0},
                "customer": {"conditions": [], "count": 0},
                "timestamp": datetime.now().isoformat()
            }
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _load_data(self) -> list:
        """Load existing data from single file"""
        try:
            with open("offers.json", "r") as file:
                data = json.load(file)
                # Initialize rules from last offer if available
                if data:
                    st.session_state.rules = data[-1].get("rules", {
                        "merchant": {"conditions": [], "count": 0},
                        "transaction": {"conditions": [], "count": 0},
                        "customer": {"conditions": [], "count": 0}
                    })
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_data(self):
        """Save all data to single file"""
        with open("offers.json", "w") as file:
            json.dump(st.session_state.offers, file, indent=4)

    def get_count(self, category: str, condition: str) -> str:
        """Execute query and return count based on the category"""
        table_mapping = {
            "merchant": "merchants",
            "transaction": "transactions",
            "customer": "customers"
        }
        
        table_name = table_mapping.get(category)
        if not table_name:
            return "Error: Invalid category"
        
        try:
            if not condition.strip():
                return "Error: Condition cannot be empty"
            
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {condition}"
            query_result = self.agent.invoke({"input": query})
            
            if isinstance(query_result, dict) and "output" in query_result:
                return query_result["output"]
            return "Error: Invalid query result"
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            return "Error"

    def add_message(self, role: str, content: str):
        """Add message to chat history"""
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.markdown(content)
    
    def handle_category_flow(self, category: str):
        """Manage category-specific rule creation flow"""
        if st.session_state.current_step == f"{category}_start":
            self.prompt_category_decision(category)
        elif st.session_state.current_step == f"{category}_condition":
            self.handle_condition_input(category)

    def prompt_category_decision(self, category: str):
        """Prompt user to create rules for a category"""
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != f"Do you want to create a {category} rule? (yes/no)":
            self.add_message("assistant", f"Do you want to create a {category} rule? (yes/no)")
        
        if user_input := st.chat_input("Type yes/no..."):
            self.add_message("user", user_input)
            
            if user_input.lower() == "yes":
                st.session_state.current_step = f"{category}_condition"
            else:
                self.move_to_next_category(category)
            st.rerun()

    def handle_condition_input(self, category: str):
        """Handle condition input for a category"""
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != f"Write {category} condition (e.g., '{self.get_example_condition(category)}')":
            self.add_message("assistant", f"Write {category} condition (e.g., '{self.get_example_condition(category)}')")
        
        if condition_input := st.chat_input("Type condition..."):
            self.add_message("user", condition_input)
            
            count = self.get_count(category, condition_input)
            self.add_message("assistant", f"Matching {category}: {count}")
            
            st.session_state.rules[category]["conditions"].append({
                "condition": condition_input,
                "count": count,
                "timestamp": datetime.now().isoformat()
            })
            st.session_state.rules[category]["count"] = count
            self.move_to_next_category(category)
            st.rerun()

    def move_to_next_category(self, current_category: str):
        """Progress to next category or start offer creation"""
        category_order = ["merchant", "transaction", "customer"]
        current_index = category_order.index(current_category)
        
        if current_index < len(category_order) - 1:
            st.session_state.current_step = f"{category_order[current_index + 1]}_start"
        else:
            st.session_state.current_step = "offer_start"

    def handle_offer_creation(self):
        """Manage offer creation workflow"""
        if st.session_state.current_step == "offer_start":
            self.prompt_offer_decision()
        elif st.session_state.current_step == "offer_details":
            self.collect_offer_details()

    def prompt_offer_decision(self):
        """Prompt user to create offers"""
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != "Create a new offer based on these rules? (yes/no)":
            self.add_message("assistant", "Create a new offer based on these rules? (yes/no)")
        
        if user_input := st.chat_input("Type yes/no..."):
            self.add_message("user", user_input)
            
            if user_input.lower() == "yes":
                st.session_state.current_step = "offer_details"
                st.session_state.current_offer = {}
                st.session_state.offer_step = 0
                st.rerun()
            else:
                st.session_state.current_step = "complete"
                st.rerun()

    def collect_offer_details(self):
        """Collect offer details from user"""
        steps = [
            ("name", "Enter offer name:"),
            ("discount", "Enter discount percentage:"),
            ("description", "Enter offer description:"),
            ("validity", "Enter offer validity (YYYY-MM-DD):")
        ]
        
        current_step = st.session_state.offer_step
        
        if current_step < len(steps):
            field, prompt = steps[current_step]
            
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != prompt:
                self.add_message("assistant", prompt)
            
            if user_input := st.chat_input("Type..."):
                self.add_message("user", user_input)
                st.session_state.current_offer[field] = user_input
                st.session_state.offer_step += 1
                st.rerun()
        else:
            # Save complete offer with current rules
            st.session_state.offers.append({
                **st.session_state.current_offer,
                "created_at": datetime.now().isoformat(),
                "rules": st.session_state.rules.copy()
            })
            self.save_data()
            
            # Reset for potential new offer
            st.session_state.current_step = "offer_start"
            st.session_state.current_offer = {}
            st.session_state.offer_step = 0
            st.rerun()

    @staticmethod
    def get_example_condition(category: str) -> str:
        """Get example SQL conditions"""
        examples = {
            "merchant": "location = 'Cairo' AND revenue > 10000",
            "transaction": "amount > 100 AND timestamp >= '2023-01-01'",
            "customer": "age > 30 AND location = 'New York'"
        }
        return examples.get(category, "")

def main():
    st.title("💬 Business Rules & Offers System")
    st.markdown("Create targeted offers based on business rules")
    
    system = QuerySystem()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Main workflow control
    if st.session_state.current_step == "start":
        st.session_state.current_step = "merchant_start"
        st.rerun()
        
    elif st.session_state.current_step == "complete":
        # Show final summary
        summary = ["## Final Offer Summary"]
        if st.session_state.offers:
            latest_offer = st.session_state.offers[-1]
            summary.append(f"### {latest_offer['name']}")
            summary.append(f"**Discount**: {latest_offer['discount']}%")
            summary.append(f"**Description**: {latest_offer['description']}")
            summary.append(f"**Valid Until**: {latest_offer['validity']}")
            
            summary.append("### Business Rules")
            for category in ["merchant", "transaction", "customer"]:
                if latest_offer["rules"][category]["conditions"]:
                    summary.append(f"#### {category.title()} Rules")
                    for idx, rule in enumerate(latest_offer["rules"][category]["conditions"], 1):
                        summary.append(
                            f"{idx}. {rule['condition']}\n"
                            f"   - Matches: {rule['count']}\n"
                            f"   - Created: {rule['timestamp']}"
                        )
        
        system.add_message("assistant", "\n".join(summary))
        
    elif "offer" in st.session_state.current_step:
        system.handle_offer_creation()
        
    else:
        current_category = st.session_state.current_step.split('_')[0]
        if current_category in ["merchant", "transaction", "customer"]:
            system.handle_category_flow(current_category)

    # Sidebar components
    with st.sidebar:
        st.header("🗃️ Database Schema")
        try:
            inspector = inspect(system.engine)
            tables = inspector.get_table_names()
            
            for table in tables:
                with st.expander(table):
                    columns = inspector.get_columns(table)
                    for col in columns:
                        st.markdown(f"- `{col['name']}` ({col['type']})")
        except Exception as e:
            st.error(f"Schema loading error: {str(e)}")

        st.divider()
        st.header("📈 Activity Summary")
        st.markdown(f"**Total Offers Created**: {len(st.session_state.offers)}")
        
        st.divider()
        if st.button("🔄 Reset System", help="Clear all data and restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()