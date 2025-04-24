import asyncio
import json
from aiohttp import web

async def human(request):
    try:
        params = await request.json()
        
        sessionid = params.get('sessionid',0)
        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()
            
        # 檢查是否有編碼指定
        text = params['text']
        encoding = params.get('encoding', '')
        
        if encoding == 'base64':
            import base64
            text = base64.b64decode(text).decode('utf-8')
        elif encoding == 'url':
            import urllib.parse
            text = urllib.parse.unquote(text)
        elif encoding == 'hex':
            import binascii
            text = binascii.unhexlify(text).decode('utf-8')
            
        if params['type']=='echo':
            nerfreals[sessionid].put_msg_txt(text)
        elif params['type']=='chat':
            res=await asyncio.get_event_loop().run_in_executor(None, llm_response, text, nerfreals[sessionid])
            
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "data":"ok"}
            ),
        )
    except Exception as e:
        logger.info(f"Error in human endpoint: {str(e)}")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "error": str(e)}
            ),
        ) 