import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from game_agent.agent import Assessment, build_agent

ROOT=Path(__file__).resolve().parents[1]
class Model:
    def __init__(self,outcome):self.outcome=outcome
    def with_structured_output(self,schema):return self
    def invoke(self,messages):
        payload=json.loads(messages[1].content)
        assert payload['answer']=='Наш свободный ответ'
        return Assessment(feedback='Обсудите план',strengths=[],risks=[],outcome=self.outcome,follow_up='Кто отвечает за звонок?' if self.outcome=='needs_clarification' else '')

@pytest.mark.parametrize('outcome,action',[('balanced','continue'),('risky','continue'),('needs_clarification','clarify')])
def test_graph_routes_free_answers(outcome,action):
    result=build_agent(Model(outcome)).invoke({'author_text':'История','answer':'Наш свободный ответ','elapsed_seconds':3,'history':[]})
    assert result['next_action']==action

@pytest.mark.skipif(not (ROOT/'data/author_text.json').exists(), reason='Полные исходные материалы поставляются только в локальном архиве')
def test_author_text_is_preserved():
    author=json.loads((ROOT/'data/author_text.json').read_text())
    cases=json.loads((ROOT/'data/scenarios.json').read_text())
    assert len(author['document'])==98
    for case in cases:
        for stage in case['stages']:
            for line in stage.splitlines():assert line in author['document']

def test_default_entry_opens_persistent_group_game():
    app=AppTest.from_file(str(ROOT/'app.py')).run()
    assert not app.exception
    assert any(button.label == 'Создать игру' for button in app.button)
    assert any(area.label.startswith('Участники') for area in app.text_area)
