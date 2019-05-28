import bs4
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

from selenium.webdriver.firefox.options import Options

def get_info(name):
    base_url = 'https://www.nemlig.com/'
    browser = webdriver.Firefox()
    # browser = webdriver.PhantomJS()

    # options = Options()
    # options.add_argument('--headless')
    # browser = webdriver.Firefox(options=options)

    browser.get(base_url)
    browser.implicitly_wait(3)

    search_field = browser.find_element_by_tag_name('input')
    search_field.send_keys(name)
    search_field.submit()

    # sleep(2)
    # browser.implicitly_wait(3)

    select = Select(browser.find_element_by_id('filter-sorting'))

    # select by visible text
    select.select_by_visible_text('Billigst')
    print(browser.current_url)

    browser.find_element_by_xpath("//a[@class='productlist-item__link']").click()

    sleep(1)

    page_source = browser.page_source

    soup = bs4.BeautifulSoup(page_source, 'html.parser')
    price_cells = soup.find_all('div', {'class':'pricecontainer__base-price'})
    name_cells = soup.find_all('div', {'class':'product-detail__info'})
    
    # print(name_cells)
    product = name_cells[0].select("h1")[0].getText()
    price = price_cells[0].select("span")[0].getText()
    decimals = price_cells[0].select("sup")[0].getText()

    # print(f"Price: {price},{decimals}")
    print(f"Product: {product} Price: {price},{decimals}")
    print(browser.current_url)

    browser.close()

if __name__ == "__main__":
    # for x in range(3):
    #     get_info('Banan')
    get_info('Banan')
    get_info('Æble')