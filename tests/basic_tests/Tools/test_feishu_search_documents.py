from unittest.mock import Mock

import pytest
import lazyllm
from lazyllm.common import Credential
from lazyllm.tools.fs.supplier.feishu import FeishuFS, FeishuWikiFS
from lazyllm.tools.agent.toolsManager import ToolManager


@pytest.fixture(params=[FeishuFS, FeishuWikiFS])
def fs(request):
    lazyllm.init_session()
    lazyllm.locals['_lazyllm_agent'] = {'workspace': {}}
    instance = request.param(dynamic_auth=True, skip_instance_cache=True)
    lazyllm.globals.config['dynamic_fs_auth'] = {instance._fs_protocol_key: 'user-token'}
    return instance


def test_global_search_preserves_provider_data_and_paging(fs):
    item = {'entity_type': 'doc', 'summary_highlighted': 'timeout is 30s',
            'result_meta': {'url': 'https://example.feishu.cn/docx/x'}}
    fs._post = Mock(return_value={'code': 0, 'data': {
        'res_units': [item], 'has_more': True, 'page_token': 'next', 'total': 21,
    }})
    result = fs.search_documents(' plan ', page_size=2, page_token='previous')
    assert result['results'] == [item]
    assert result['has_more'] and result['page_token'] == 'next'
    fs._post.assert_called_once_with(fs._base_url + '/search/v2/doc_wiki/search',
                                     json={'query': 'plan', 'page_size': 2, 'page_token': 'previous',
                                           'doc_filter': {}, 'wiki_filter': {}}, timeout=15)


def test_missing_user_and_tenant_credentials_do_not_send(fs):
    fs._post = Mock()
    lazyllm.globals.config['dynamic_fs_auth'] = {}
    with pytest.raises(ValueError):
        fs.search_documents('plan')
    fs._credential = Credential(kind='app_credentials', access_token='tenant-token')
    with pytest.raises(ValueError, match='user OAuth'):
        fs.search_documents('plan')
    fs._post.assert_not_called()


def test_search_preserves_authorization_header(fs):
    response = Mock(ok=True, status_code=200, content=b'json')
    response.json.return_value = {'code': 0, 'data': {'res_units': []}}
    fs._session.request = Mock(return_value=response)
    assert fs.search_documents('plan')['results'] == []
    assert fs._session.request.call_args.kwargs['headers']['Authorization'] == 'Bearer user-token'


@pytest.mark.parametrize('code', [99991663, 99991672])
def test_provider_permission_and_expiration_are_not_empty_success(fs, code):
    fs._post = Mock(return_value={'code': code, 'msg': 'authorization error'})
    with pytest.raises(RuntimeError, match=str(code)):
        fs.search_documents('plan')
    fs._post.assert_called_once()


def test_tool_exposed_for_drive_and_wiki(fs):
    manager = ToolManager([fs])
    manager._tool_call[f'get_{type(fs).__name__}_methods']({})
    assert any(item['function']['name'].endswith('_search_documents') for item in manager.tools_description)


@pytest.mark.parametrize('kwargs', [{'query': ''}, {'page_size': True}, {'page_size': 51}, {'page_token': None}])
def test_input_validation(fs, kwargs):
    fs._post = Mock()
    with pytest.raises(ValueError):
        fs.search_documents(**({'query': 'plan'} | kwargs))
    fs._post.assert_not_called()


def test_product_token_injection_reaches_both_suppliers(fs):
    from lazyllm.tools.tool_config_inject import inject_tool_config
    lazyllm.globals.config['dynamic_fs_auth'] = {}
    inject_tool_config({'feishu': 'connected-user'})
    response = Mock(ok=True, status_code=200, content=b'json')
    response.json.return_value = {'code': 0, 'data': None}
    fs._session.request = Mock(return_value=response)
    result = fs.search_documents('plan')
    assert result['results'] == [] and not result['has_more']
    assert fs._session.request.call_args.kwargs['headers']['Authorization'] == 'Bearer connected-user'


@pytest.mark.parametrize('status', [401, 403])
def test_http_auth_error_propagates_without_fallback(fs, status):
    from lazyllm.common import KeyAuthError
    fs._session.request = Mock(return_value=Mock(status_code=status, ok=False))
    with pytest.raises(KeyAuthError):
        fs.search_documents('plan')
    fs._session.request.assert_called_once()
