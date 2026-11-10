#!/usr/bin/env python3
"""Build the report by patching a copy of the retained DOCX package.

Use the bundled Codex Python runtime. No model dependencies are loaded.
Source content is report_content.py. Rendering is a separate QA step.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QA = HERE / 'writing_qa'
QA.mkdir(exist_ok=True)
REFERENCE = ROOT / 'SampleReport.docx'
FINAL = HERE / 'Refusal_Direction_Ablation_Report.docx'
EXPECTED = 'e62bd2c6c049f854c84896c9ff903e4a04d583ca25ec51399f9895474e260f1a'
assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED

sys.path.insert(0, str(HERE))
from report_content import TITLE, SUBTITLE, ABSTRACT, KEYWORDS, PAGES, REFERENCES, APPENDIX_PAGES

SKILL = Path('/home/op/.codex/plugins/cache/openai-primary-runtime/documents/26.1105.12137/skills/documents')
sys.path.insert(0, str(SKILL / 'scripts'))
from docx_ooxml_patch import _qn as helper_qn

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PR = 'http://schemas.openxmlformats.org/package/2006/relationships'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
NS = {'w':W, 'r':R}
def q(n): return helper_qn('w', n)
def node(tag, **attrs):
    e=E.Element(q(tag))
    for k,v in attrs.items(): e.set(q(k),str(v))
    return e
def xml_bytes(e): return E.tostring(e,encoding='UTF-8',xml_declaration=True,standalone=True)

parts={}
with zipfile.ZipFile(REFERENCE) as z:
    infos=z.infolist()
    for info in infos: parts[info.filename]=z.read(info.filename)
doc=E.fromstring(parts['word/document.xml'])
body=doc.find('w:body',NS)
original_sections=[copy.deepcopy(s) for s in doc.findall('.//w:sectPr',NS)]
assert len(original_sections)==18
for child in list(body): body.remove(child)

# Citation order follows the abstract and substantive content, never generated list pages.
citation_order=[]
def cite_text(text):
    def repl(m):
        key=m.group(1)
        if key not in REFERENCES: raise ValueError(key)
        if key not in citation_order: citation_order.append(key)
        return '['+str(citation_order.index(key)+1)+']'
    return re.sub(r'\{([a-z]+)\}',repl,text)
ABSTRACT=cite_text(ABSTRACT)
for pg in PAGES+APPENDIX_PAGES:
    for b in pg['blocks']:
        for field in ['text','caption']:
            if field in b:b[field]=cite_text(b[field])
        if 'headers' in b:b['headers']=[cite_text(str(v)) for v in b['headers']]
        if 'rows' in b:
            b['rows']=[[cite_text(str(v)) for v in row] if isinstance(row,list) else cite_text(str(row)) for row in b['rows']]
assert len(citation_order)==12, citation_order

def run(text, *, size=12,bold=False,italic=False,font='Times New Roman'):
    r=node('r');rp=node('rPr')
    rp.append(node('rFonts',ascii=font,hAnsi=font,eastAsia=font,cs=font))
    rp.append(node('color',val='000000'))
    rp.append(node('sz',val=int(size*2)));rp.append(node('szCs',val=int(size*2)))
    if bold: rp.append(node('b'))
    if italic: rp.append(node('i'))
    r.append(rp)
    for i,line in enumerate(text.split('\n')):
        if i:r.append(node('br'))
        t=node('t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=line;r.append(t)
    return r

# Native OMML keeps the few equations editable and gives subscripts, fractions
# and transposes their proper structure. The final rendered pages are the
# cross-renderer verification for these standard OMML constructs.
def mn(tag, *children, **attrs):
    el=E.Element('{'+M+'}'+tag)
    for key,value in attrs.items():el.set('{'+M+'}'+key,str(value))
    for child in children:el.append(child)
    return el

def mr(text, normal=False):
    el=mn('r')
    if normal:el.append(mn('rPr',mn('nor')))
    props=node('rPr');props.append(node('rFonts',ascii='Cambria Math',hAnsi='Cambria Math'))
    props.append(node('sz',val=23));el.append(props)
    t=mn('t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=text;el.append(t)
    return el

def me(*children):return mn('e',*children)
def sub(base,index):return mn('sSub',me(mr(base)),mn('sub',mr(index)))
def sup(base,index):return mn('sSup',me(base if hasattr(base,'tag') else mr(base)),mn('sup',mr(index)))
def frac(numerator,denominator):
    return mn('f',mn('fPr',mn('type',val='lin')),mn('num',*numerator),mn('den',*denominator))
def root(*children):return mn('rad',mn('radPr',mn('degHide',val=1)),mn('deg'),me(*children))
def norm(*children):return [mr('‖'),*children,mr('‖')]

def inline_math(token):
    base,index=token.split('_')
    return mn('oMath',sub(base,index))

def add_mixed(p,text,**kwargs):
    for segment in re.split(r'(μ_[hb]|u_b|d_k|p_[WD])',text):
        if not segment:continue
        if re.fullmatch(r'μ_[hb]|u_b|d_k|p_[WD]',segment):p.append(inline_math(segment))
        else:p.append(run(segment,**kwargs))

def equation_xml(number):
    if number=='2.1':
        contents=[mr('Attention',True),mr('(Q, K, V) = '),mr('softmax',True),mr('('),
                  frac([mr('Q'),sup('K','T')],[root(sub('d','k'))]),mr(') V')]
    elif number=='3.1':
        first=[mr('d = '),sub('μ','h'),mr(' − '),sub('μ','b'),mr(';  '),sub('u','b'),mr(' = '),
               frac([sub('μ','b')],norm(sub('μ','b')))]
        second=[sub('d','⊥'),mr(' = d − '),sub('u','b'),mr('('),sup(sub('u','b'),'T'),mr('d);  r = '),
                frac([sub('d','⊥')],norm(sub('d','⊥')))]
        contents=[mn('eqArr',mn('eqArrPr',mn('baseJc',val='center')),me(*first),me(*second))]
    elif number=='3.2':
        contents=[mr('M′ = M − λr('),sup('r','T'),mr('M) = M + BA')]
    elif number=='3.3':
        contents=[sub('KL','1'),mr(' = '),frac([mr('1')],[mr('N')]),sub('∑','x'),sub('∑','v'),
                  sub('p','W'),mr('(v|x) '),mr('log',True),mr('['),
                  frac([sub('p','W'),mr('(v|x)')],[sub('p','D'),mr('(v|x)')]),mr(']')]
    else:raise ValueError(number)
    p=paragraph('',align='center',first=0,left=0,right=0,size=11.5,
                line=310,before=60,after=160)
    p.append(mn('oMath',*contents));p.append(run('    ('+number+')',size=11.5,font='Cambria Math'))
    return p

bookmark_id=100
def paragraph(text='', *, style='BodyText', align='both', size=12,bold=False,italic=False,
              before=0, after=120, line=326, first=432, left=142,right=707,keep=False,
              pagebreak=False, anchor=None, font='Times New Roman'):
    global bookmark_id
    p=node('p');pp=node('pPr');p.append(pp)
    pp.append(node('pStyle',val=style))
    pp.append(node('spacing',before=before,after=after,line=line,lineRule='auto'))
    pp.append(node('ind',left=left,right=right,firstLine=first))
    pp.append(node('jc',val=align));pp.append(node('widowControl'))
    if keep:pp.append(node('keepNext'))
    if pagebreak:pp.append(node('pageBreakBefore'))
    if anchor:
        bookmark_id+=1
        p.append(node('bookmarkStart',id=bookmark_id,name=anchor))
    if text:add_mixed(p,text,size=size,bold=bold,italic=italic,font=font)
    if anchor:p.append(node('bookmarkEnd',id=bookmark_id))
    return p

def h1(text, anchor=None, before=0):
    return paragraph(text,style='Heading1',align='center',size=14,bold=True,before=before,
                     after=240,line=300,first=0,left=0,right=0,keep=True,anchor=anchor)

def h2(text):
    return paragraph(text,style='Heading2',align='left',size=12,bold=True,before=120,
                     after=120,line=300,first=0,left=142,right=707,keep=True)

def table_xml(block):
    caption=paragraph(block['caption'],align='center',size=11,bold=True,after=100,
                      first=0,left=0,right=0,line=280,keep=True)
    table=node('tbl');pr=node('tblPr')
    widths=[int(w*1440) for w in block['widths']]
    pr.append(node('tblW',w=sum(widths),type='dxa'))
    pr.append(node('jc',val='center'))
    pr.append(node('tblLayout',type='fixed'))
    borders=node('tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        borders.append(node(edge,val='single',sz=4,color='D9D9D9'))
    pr.append(borders)
    margins=node('tblCellMar')
    for edge,val in [('top',90),('bottom',90),('left',100),('right',100)]:
        margins.append(node(edge,w=val,type='dxa'))
    pr.append(margins);table.append(pr)
    grid=node('tblGrid')
    for w in widths:grid.append(node('gridCol',w=w))
    table.append(grid)
    for j,row in enumerate([block['headers']]+block['rows']):
        tr=node('tr');trpr=node('trPr');trpr.append(node('cantSplit'))
        if j==0:trpr.append(node('tblHeader'))
        tr.append(trpr)
        for k,text in enumerate(row):
            tc=node('tc');tcpr=node('tcPr');tcpr.append(node('tcW',w=widths[k],type='dxa'))
            tcpr.append(node('vAlign',val='center'))
            if j==0:tcpr.append(node('shd',val='clear',color='auto',fill='EEEEEE'))
            tc.append(tcpr)
            numeric=(k>0 and len(str(text))<18 and not any(c.isalpha() for c in str(text)))
            tc.append(paragraph(str(text),align='center' if numeric or j==0 else 'left',
                    size=10.5,bold=j==0,first=0,left=0,right=0,after=20,line=270))
            tr.append(tc)
        table.append(tr)
    return [caption,table,paragraph('',after=70,line=120,first=0,left=0,right=0,size=1)]

def workflow_image():
    # Study logic only; arrows make no claim of causal or measured effects.
    im=Image.new('RGB',(1800,690),'white');d=ImageDraw.Draw(im)
    fpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fpath,37)
    def box(x,y,w,h,lines):
        d.rectangle((x,y,x+w,y+h),outline='black',width=3)
        height=len(lines)*49
        for j,s in enumerate(lines):
            b=d.textbbox((0,0),s,font=font);tw=b[2]-b[0]
            d.text((x+(w-tw)/2,y+(h-height)/2+j*49),s,font=font,fill='black')
    def down(x,y1,y2):
        d.line((x,y1,x,y2),fill='black',width=3)
        d.polygon([(x,y2),(x-10,y2-18),(x+10,y2-18)],fill='black')
    box(30,30,530,120,['Refusal control'])
    box(635,30,530,120,['Semantic evaluation'])
    box(1240,30,530,120,['Quantization safety'])
    for x in [295,900,1505]:
        d.line((x,150,x,210),fill='black',width=3)
    d.line((295,210,1505,210),fill='black',width=3)
    down(900,210,265)
    box(170,265,1460,125,['Does a fixed edit retain its intended effect','after compression?'])
    down(900,390,465)
    box(110,465,1580,180,['Floating controls: original W, publisher P, reconstruction D',
        'NF4: Q(W), Q(W + Δ), and Q(W) + frozen Δ'])
    im.save(QA/'study_workflow.png',dpi=(300,300))
workflow_image()

image_rel='rIdReportFigure1'
def figure_xml(block):
    p=paragraph('',align='center',first=0,left=0,right=0,after=60,keep=True)
    drawing=E.fromstring(f'''<w:r xmlns:w="{W}" xmlns:r="{R}" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="5486400" cy="2103120"/><wp:docPr id="501" name="Study components" descr="Three literature themes motivate a fixed edit stability question, tested with floating controls and three native NF4 constructions; arrows express study logic"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="study_workflow.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{image_rel}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="5486400" cy="2103120"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>''')
    p.append(drawing)
    c=paragraph(block['caption'],align='center',size=11,first=0,left=0,right=0,
                line=280,after=140)
    return [p,c]

def render_block(b):
    t=b['type']
    if t=='p':return [paragraph(b['text'],align=b.get('align','both'))]
    if t=='h':return [h2(b['text'])]
    if t=='table':return table_xml(b)
    if t=='figure':return figure_xml(b)
    if t=='equation':
        return [equation_xml(b['number'])]
    if t=='steps':
        out=[paragraph(b['caption'],align='left',bold=True,size=11.5,first=0,left=142,right=707,keep=True)]
        for i,s in enumerate(b['rows']):
            out.append(paragraph(str(i+1)+'. '+s,align='left',first=0,left=300,right=707,
                                 size=11.5,line=290,after=60,keep=i<len(b['rows'])-1))
        return out
    raise ValueError(t)

def entry(text,page_number,anchor):
    p=paragraph('',style='TOC1',align='left',first=0,left=142,right=150,after=160,line=290)
    pp=p.find('w:pPr',NS);tabs=node('tabs');tabs.append(node('tab',val='right',leader='dot',pos=9600));pp.append(tabs)
    link=E.SubElement(p,q('hyperlink'));link.set(q('anchor'),'pg_'+anchor)
    link.append(run(text,size=12,bold=False))
    r=node('r');r.append(node('tab'));p.append(r);p.append(run(str(page_number),size=12))
    return p

front=[]
def fp(key,title,blocks,section):front.append({'key':key,'title':title,'blocks':blocks,'section':section,'front':True})
fp('cover','',[],0)
fp('approval','APPROVAL OF THE VIVA VOCE BOARD',[],1)
fp('ack','ACKNOWLEDGEMENTS',[],2)
fp('abstract','ABSTRACT',[],3)
fp('contents','TABLE OF CONTENTS',[],4)
fp('tables','LIST OF TABLES',[],5)
fp('figures','LIST OF FIGURES',[],6)
fp('algorithms','LIST OF ALGORITHMS',[],7)
fp('abbreviations','LIST OF ABBREVIATIONS',[],8)
ref_pages=[{'key':'references1','title':'REFERENCES','blocks':[],'section':15},
           {'key':'references2','title':'','blocks':[],'section':16}]
pages=front+PAGES+ref_pages+APPENDIX_PAGES
assert len(pages)==26
page_map={pg['key']:i+1 for i,pg in enumerate(pages)}
logical={key:(n-9 if n>=10 else ['', '', 'i','ii','iii','iv','v','vi','vii','viii'][n]) for key,n in page_map.items()}

table_entries=[];fig_entries=[];alg_entries=[]
for pg in PAGES+APPENDIX_PAGES:
    for b in pg['blocks']:
        for typ,target in [('table',table_entries),('figure',fig_entries),('steps',alg_entries)]:
            if b['type']==typ: target.append((b['caption'],logical[pg['key']],pg['key']))

def front_elements(pg):
    key=pg['key'];out=[]
    if key=='cover':
        out=[paragraph(TITLE,style='Title',align='center',size=20.5,bold=True,first=0,left=0,right=0,line=290,after=200),
             paragraph(SUBTITLE,align='center',size=13,first=0,left=0,right=0,after=380,line=290),
             paragraph('A Technical Writing Report Submitted',align='center',first=0,left=0,right=0,after=0),
             paragraph('in partial fulfillment of the requirements for the award of the',align='center',first=0,left=0,right=0,after=280),
             paragraph('Bachelor of Technology',style='Title',align='center',size=20.5,bold=True,first=0,left=0,right=0,after=220),
             paragraph('by',align='center',first=0,left=0,right=0,after=100),
             paragraph('[Student Name]',align='center',size=14,bold=True,first=0,left=0,right=0,after=0),
             paragraph('[Student ID]',align='center',size=12,first=0,left=0,right=0,after=700),
             paragraph('Under the supervision of',align='center',first=0,left=0,right=0,after=0),
             paragraph('[Supervisor Name and Title]',align='center',size=14,bold=True,first=0,left=0,right=0,after=900),
             paragraph('[Institute Emblem]',align='center',first=0,left=0,right=0,after=350),
             paragraph('[Department Name]',align='center',first=0,left=0,right=0,after=0),
             paragraph('[Institute Name]',align='center',first=0,left=0,right=0,after=160),
             paragraph('November 2026',align='center',first=0,left=0,right=0)]
    elif key=='approval':
        out=[h1(pg['title']),paragraph('Examination date  [To be completed]',align='right',first=0,left=0,right=0,before=400,after=300),
             paragraph('The report entitled “'+TITLE.replace('\n',' ')+'” is submitted by [Student Name] ([Student ID]) to [Institute Name] in partial fulfillment of the requirements for the award of the Bachelor of Technology degree in [Programme Name].'),
             paragraph('The viva voce board may record its decision below after examining the report and the student. Approval, examination dates and signatures are to be completed by the authorized examiners.'),
             paragraph('Decision  ______________________________________________',align='left',first=0,left=142,right=707,before=300,after=700),
             paragraph('____________________                         ____________________',align='center',first=0,left=0,right=0,after=60),
             paragraph('Supervisor                                                   Panel Members',align='center',first=0,left=0,right=0,bold=True)]
    elif key=='ack':
        out=[h1(pg['title']),paragraph('The authors and maintainers of the model, software and evaluation resources used in this study are acknowledged through the references and provenance appendix. Their published artifacts made the reconstruction and comparison possible.',before=400),
             paragraph('[The student may add personal acknowledgements here after confirming the names and contributions of the supervisor, faculty members and other contributors.]'),
             paragraph('[Student Name]  [Student ID]',align='right',first=0,left=0,right=0,before=500)]
    elif key=='abstract':
        out=[h1(pg['title'])]+[paragraph(s) for s in ABSTRACT.split('\n\n')]
        out.append(paragraph('Keywords  '+KEYWORDS,first=0,left=142,right=707,bold=False,before=120))
    elif key=='contents':
        out=[h1(pg['title'])]
        contents=[('ABSTRACT','abstract'),('TABLE OF CONTENTS','contents'),('LIST OF TABLES','tables'),
                  ('LIST OF FIGURES','figures'),('LIST OF ALGORITHMS','algorithms'),('LIST OF ABBREVIATIONS','abbreviations'),
                  ('1  INTRODUCTION','intro1'),('2  RELATED WORK','related1'),('3  METHODOLOGY','method1'),
                  ('4  RESULTS AND DISCUSSION','results1'),('5  CONCLUSION AND FUTURE SCOPE','conclusion'),
                  ('REFERENCES','references1'),('APPENDIX A  REPRODUCIBILITY DETAILS','appendix1')]
        for text,k in contents:out.append(entry(text,logical[k],k))
    elif key in ['tables','figures','algorithms']:
        out=[h1(pg['title'])]
        for caption,n,anchor in {'tables':table_entries,'figures':fig_entries,'algorithms':alg_entries}[key]:
            out.append(entry(caption,n,anchor))
    elif key=='abbreviations':
        out=[h1(pg['title'])]
        terms=[('EOS','End of sequence'),('FP16','Sixteen bit floating point'),('FP32','Thirty two bit floating point'),
               ('GPU','Graphics processing unit'),('KL','Kullback Leibler divergence'),('LLM','Large language model'),
               ('LoRA','Low rank adaptation'),('MLP','Multilayer perceptron'),('NF4','Four bit NormalFloat'),
               ('PTQ','Post training quantization'),('RQ','Research question'),('SHA','Secure hash algorithm')]
        for a,b in terms:out.append(paragraph(a+'   '+b,align='left',first=0,left=142,right=707,after=170))
    return out

def refs_elements(index):
    out=[h1('REFERENCES')] if index==0 else [h2('References continued')]
    selected=citation_order[index*6:(index+1)*6]
    for key in selected:
        ref=REFERENCES[key];num=citation_order.index(key)+1
        pp=paragraph(f'[{num}] '+ref['text']+'\n[Online]. Available: '+ref['url'],align='left',first=0,left=500,right=250,
                     size=11,line=290,after=220,keep=True)
        ind=pp.find('w:pPr/w:ind',NS);ind.attrib.pop(q('firstLine'),None);ind.set(q('hanging'),'400')
        out.append(pp)
    return out

def section_properties(index):
    # Reuse the supplied page patterns; eliminate unused continuous/two-column fragments.
    source_index=0 if index==0 else (1 if index<4 else 13)
    sect=copy.deepcopy(original_sections[source_index])
    for x in list(sect):
        if E.QName(x).localname in ['footerReference','headerReference','pgNumType','type']:
            sect.remove(x)
    if index:
        ref=node('footerReference',type='default');ref.set('{'+R+'}id','rId8' if index<9 else 'rId9');sect.insert(0,ref)
    sect.insert(1 if index else 0,node('type',val='nextPage'))
    if index==1:sect.append(node('pgNumType',fmt='lowerRoman',start=1))
    elif 1<index<9:sect.append(node('pgNumType',fmt='lowerRoman'))
    elif index==9:sect.append(node('pgNumType',fmt='decimal',start=1))
    elif index>9:sect.append(node('pgNumType',fmt='decimal'))
    return sect

for i,pg in enumerate(pages):
    if pg.get('front'): elements=front_elements(pg)
    elif pg['key'].startswith('references'): elements=refs_elements(0 if pg['key']=='references1' else 1)
    else:
        elements=[]
        if pg.get('chapter'):elements.append(h1('CHAPTER '+str(pg['chapter'])))
        if pg['title']:elements.append(h1(pg['title']))
        for block in pg['blocks']:
            if pg['key'].startswith('appendix') and block['type']=='p':
                block={**block,'align':'left'}
            elements.extend(render_block(block))
    first=next(e for e in elements if e.tag==q('p'))
    # A stable anchor on every page also supports the static contents/list links.
    bookmark_id+=1
    first.insert(1,node('bookmarkStart',id=bookmark_id,name='pg_'+pg['key']))
    first.append(node('bookmarkEnd',id=bookmark_id))
    if i and pages[i-1]['section']==pg['section']:
        first.find('w:pPr',NS).append(node('pageBreakBefore'))
    for e in elements:body.append(e)
    section_ends=(i==len(pages)-1 or pages[i+1]['section']!=pg['section'])
    if section_ends:
        sect=section_properties(pg['section'])
        if i==len(pages)-1:body.append(sect)
        else:
            last=body[-1]
            if last.tag!=q('p'):
                last=paragraph('',size=1,after=0,line=20,first=0,left=0,right=0);body.append(last)
            last.find('w:pPr',NS).append(sect)

parts['word/document.xml']=xml_bytes(doc)
settings=E.fromstring(parts['word/settings.xml'])
update=settings.find('w:updateFields',NS)
if update is None:update=node('updateFields',val='true');settings.append(update)
else:update.set(q('val'),'true')
parts['word/settings.xml']=xml_bytes(settings)

rels=E.fromstring(parts['word/_rels/document.xml.rels'])
rel=E.SubElement(rels,'{'+PR+'}Relationship')
rel.set('Id',image_rel);rel.set('Type',R+'/image');rel.set('Target','media/report_workflow.png')
parts['word/_rels/document.xml.rels']=xml_bytes(rels)
parts['word/media/report_workflow.png']=(QA/'study_workflow.png').read_bytes()
ct=E.fromstring(parts['[Content_Types].xml'])
CT='http://schemas.openxmlformats.org/package/2006/content-types'
if not any(x.get('Extension')=='png' for x in ct):
    e=E.SubElement(ct,'{'+CT+'}Default');e.set('Extension','png');e.set('ContentType','image/png')
parts['[Content_Types].xml']=xml_bytes(ct)

core=E.fromstring(parts['docProps/core.xml'])
for name,value in [('title',TITLE.replace('\n',' ')),('creator','[Student Name]')]:
    tag='{http://purl.org/dc/elements/1.1/}'+name
    e=core.find(tag)
    if e is None:e=E.SubElement(core,tag)
    e.text=value
last=core.find('{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy')
if last is not None:last.text='[Student Name]'
parts['docProps/core.xml']=xml_bytes(core)

with zipfile.ZipFile(FINAL,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for info in infos:z.writestr(info,parts[info.filename])
    z.writestr('word/media/report_workflow.png',parts['word/media/report_workflow.png'])

allowed={'word/document.xml','word/settings.xml','word/_rels/document.xml.rels','[Content_Types].xml','docProps/core.xml'}
with zipfile.ZipFile(REFERENCE) as z:
    changed=[name for name in z.namelist() if z.read(name)!=parts[name]]
    assert set(changed)<=allowed,changed
assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==EXPECTED
inventory={'reference_sha256':EXPECTED,'final_sha256':hashlib.sha256(FINAL.read_bytes()).hexdigest(),
           'changed_existing_parts':changed,'added_parts':['word/media/report_workflow.png'],
           'preserved_part_count':len(infos)-len(changed),'section_count':len(doc.findall('.//w:sectPr',NS)),
           'planned_physical_pages':len(pages),'planned_page_map':page_map,'logical_page_map':logical,
           'citation_order':citation_order,'abstract_words':len(ABSTRACT.split()),
           'text_words':len(' '.join(doc.xpath('//w:t/text()',namespaces=NS)).split())}
assert inventory['section_count']==18
assert 250<=inventory['abstract_words']<=300,inventory['abstract_words']
(QA/'build_manifest.json').write_text(json.dumps(inventory,indent=2))
plain=[]
for p in doc.findall('.//w:p',NS):
    t=''.join(p.xpath('.//w:t/text()',namespaces=NS))
    if t:plain.append(t)
(QA/'report_text.txt').write_text('\n\n'.join(plain))
print(json.dumps(inventory,indent=2))
